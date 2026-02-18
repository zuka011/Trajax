from typing import Final, Sequence, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
    NumPyIntegratorState,
    NumPyIntegratorStateSequence,
    NumPyIntegratorStateBatch,
    NumPyIntegratorControlInputSequence,
    NumPyIntegratorControlInputBatch,
    NumPyIntegratorObstacleStatesHistory,
    EstimatedObstacleStates,
)
from trajax.states import (
    NumPySimpleState as SimpleState,
    NumPySimpleStateSequence as SimpleStateSequence,
    NumPySimpleStateBatch as SimpleStateBatch,
    NumPySimpleControlInputBatch as SimpleControlInputBatch,
)
from trajax.filters import (
    NumPyKalmanFilter,
    NumPyGaussianBelief,
    NumPyNoiseCovarianceArrayDescription,
    NumPyNoiseCovarianceDescription,
    numpy_kalman_filter,
)
from trajax.models.common import SMALL_UNCERTAINTY, LARGE_UNCERTAINTY
from trajax.models.basic import invalid_obstacle_filter_from
from trajax.models.integrator.common import (
    observation_dimension_from,
    kf_state_dimension_for,
)

from numtypes import Array, Dims, shape_of

import numpy as np


NO_LIMITS: Final = (float("-inf"), float("inf"))

type NumPyIntegratorObstacleCovariances[D_x: int, K: int] = Array[Dims[D_x, D_x, K]]


@dataclass(frozen=True)
class NumPyIntegratorObstacleStates[D_o: int, K: int]:
    """Arbitrary obstacle states with no semantic meaning attached."""

    _array: Array[Dims[D_o, K]]

    @staticmethod
    def create(
        *, array: Array[Dims[D_o, K]]
    ) -> "NumPyIntegratorObstacleStates[D_o, K]":
        return NumPyIntegratorObstacleStates(_array=array)

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self.array

    @property
    def dimension(self) -> D_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyIntegratorObstacleStateSequences[T: int, D_x: int, K: int]:
    """Time-indexed obstacle state sequences for integrator model obstacles."""

    _array: Array[Dims[T, D_x, K]]
    _covariance: Array[Dims[T, D_x, D_x, K]]

    @staticmethod
    def create[D_x_: int, K_: int, T_: int = int](
        predictions: Sequence[NumPyGaussianBelief[D_x_, K_]],
    ) -> "NumPyIntegratorObstacleStateSequences[T_, D_x_, K_]":
        assert len(predictions) > 0, "Predictions sequence must not be empty."

        T = cast(T_, len(predictions))
        D_x = predictions[0].mean.shape[0]
        K = predictions[0].mean.shape[1]

        array = np.stack([belief.mean for belief in predictions], axis=0)
        covariance = np.stack([belief.covariance for belief in predictions], axis=0)

        assert shape_of(array, matches=(T, D_x, K))
        assert shape_of(covariance, matches=(T, D_x, D_x, K))

        return NumPyIntegratorObstacleStateSequences(array, covariance)

    def __array__(self, dtype: None | type = None) -> Array[Dims[T, D_x, K]]:
        return self.array

    def covariance(self) -> Array[Dims[T, D_x, D_x, K]]:
        return self._covariance

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_x:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_x, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyIntegratorObstacleInputs[D_o: int, K: int]:
    _array: Array[Dims[D_o, K]]

    @staticmethod
    def create(
        *, array: Array[Dims[D_o, K]]
    ) -> "NumPyIntegratorObstacleInputs[D_o, K]":
        return NumPyIntegratorObstacleInputs(_array=array)

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self.array

    def zeroed(self, *, at: tuple[int, ...]) -> "NumPyIntegratorObstacleInputs[D_o, K]":
        """Returns new obstacle inputs with inputs at specified state dimensions zeroed out."""

        zeroed_array = self.array.copy()
        zeroed_array[at, :] = 0.0

        return NumPyIntegratorObstacleInputs.create(array=zeroed_array)

    @property
    def dimension(self) -> D_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return self._array


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorModel(
    DynamicalModel[
        NumPyIntegratorState,
        NumPyIntegratorStateSequence,
        NumPyIntegratorStateBatch,
        NumPyIntegratorControlInputSequence,
        NumPyIntegratorControlInputBatch,
    ]
):
    """Point-mass model with direct position control, used for obstacle prediction."""

    _time_step_size: float
    state_limits: tuple[float, float]
    velocity_limits: tuple[float, float]
    periodic: bool

    @staticmethod
    def create(
        *,
        time_step_size: float,
        state_limits: tuple[float, float] | None = None,
        velocity_limits: tuple[float, float] | None = None,
        periodic: bool = False,
    ) -> "NumPyIntegratorModel":
        """A NumPy integrator model where state = cumulative sum of controls.

        This model represents a particle that moves according to velocity commands.
        State dimension must equal control dimension (D_x == D_u).

        $$x_{t+1} = \\text{clip}(x_t + \\text{clip}(u_t,\\; v_{\\text{lim}}) \\cdot \\Delta t,\\; s_{\\text{lim}})$$

        Args:
            time_step_size: The time step size for the integrator.
            state_limits: Optional tuple of (min, max) limits for the state values.
            velocity_limits: Optional tuple of (min, max) limits for the velocity inputs.
            periodic: Whether to apply periodic boundary conditions based on state_limits.
        """
        return NumPyIntegratorModel(
            _time_step_size=time_step_size,
            state_limits=state_limits if state_limits is not None else NO_LIMITS,
            velocity_limits=velocity_limits
            if velocity_limits is not None
            else NO_LIMITS,
            periodic=periodic,
        )

    def __post_init__(self) -> None:
        if self.periodic:
            validate_periodic_state_limits(self.state_limits)

    def simulate[T: int, D_x: int, M: int](
        self,
        inputs: NumPyIntegratorControlInputBatch[T, D_x, M],
        initial_state: NumPyIntegratorState[D_x],
    ) -> SimpleStateBatch[T, D_x, M]:
        clipped_inputs = np.clip(inputs.array, *self.velocity_limits)

        return SimpleStateBatch(
            wrap_periodic_batch(
                states=simulate(
                    inputs=clipped_inputs,
                    initial_states=initial_state.array[:, np.newaxis],
                    time_step=self.time_step_size,
                ),
                state_limits=self.state_limits,
            )
            if self.periodic
            else simulate_with_state_limits(
                inputs=clipped_inputs,
                initial_state=initial_state.array,
                time_step=self.time_step_size,
                state_limits=self.state_limits,
            )
            if self.has_state_limits
            else simulate(
                inputs=clipped_inputs,
                initial_states=initial_state.array[:, np.newaxis],
                time_step=self.time_step_size,
            )
        )

    def step[T: int, D_x: int](
        self,
        inputs: NumPyIntegratorControlInputSequence[T, D_x],
        state: NumPyIntegratorState[D_x],
    ) -> SimpleState[D_x]:
        clipped_control = np.clip(inputs.array[0], *self.velocity_limits)
        unbounded = state.array + clipped_control * self.time_step_size
        new_state = (
            wrap_periodic_state(state=unbounded, state_limits=self.state_limits)
            if self.periodic
            else np.clip(unbounded, *self.state_limits)
        )

        return SimpleState(new_state)

    def forward[T: int, D_x: int](
        self,
        inputs: NumPyIntegratorControlInputSequence[T, D_x],
        state: NumPyIntegratorState[D_x],
    ) -> SimpleStateSequence[T, D_x]:
        return self.simulate(
            inputs=SimpleControlInputBatch.of(inputs), initial_state=state
        ).rollout(0)

    @property
    def time_step_size(self) -> float:
        return self._time_step_size

    @property
    def has_state_limits(self) -> bool:
        return self.state_limits != NO_LIMITS


@dataclass(frozen=True)
class NumPyIntegratorStateEstimationModel[D_o: int, D_x: int]:
    """Single integrator model used for obstacle state estimation."""

    time_step_size: float
    observation_dimension: D_o
    initial_state_covariance: Array[Dims[D_x, D_x]]

    @staticmethod
    def create[D_o_: int, D_x_: int = int](
        *,
        time_step_size: float,
        observation_dimension: D_o_,
        initial_state_covariance: Array[Dims[D_x_, D_x_]] | None = None,
    ) -> "NumPyIntegratorStateEstimationModel[D_o_, D_x_]":
        if initial_state_covariance is None:
            initial_state_covariance = NumPyIntegratorStateEstimationModel.default_initial_state_covariance_for(
                observation_dimension
            )

        return NumPyIntegratorStateEstimationModel(
            time_step_size=time_step_size,
            observation_dimension=observation_dimension,
            initial_state_covariance=initial_state_covariance,
        )

    @staticmethod
    def default_initial_state_covariance_for[D_o_: int = int, D_x_: int = int](
        observation_dimension: D_o_,
    ) -> Array[Dims[D_x_, D_x_]]:
        D_o = observation_dimension

        # NOTE: We are sure of the observed states, unsure of the velocities.
        return np.diag(
            np.concatenate(
                (np.full(D_o, SMALL_UNCERTAINTY), np.full(D_o, LARGE_UNCERTAINTY))
            )
        )

    def __call__[K: int](
        self, augmented_state: Array[Dims[D_x, K]]
    ) -> Array[Dims[D_x, K]]:
        return self.state_transition_matrix @ augmented_state

    def states_from[K: int](
        self, belief: NumPyGaussianBelief[D_x, K]
    ) -> NumPyIntegratorObstacleStates[D_o, K]:
        D_o = self.observation_dimension
        return NumPyIntegratorObstacleStates.create(array=belief.mean[:D_o, :])

    def inputs_from[K: int](
        self, belief: NumPyGaussianBelief[D_x, K]
    ) -> NumPyIntegratorObstacleInputs[D_o, K]:
        D_o = self.observation_dimension
        return NumPyIntegratorObstacleInputs.create(array=belief.mean[D_o:, :])

    def initial_belief_from[K: int](
        self,
        *,
        states: NumPyIntegratorObstacleStates[D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
        covariances: NumPyIntegratorObstacleCovariances[D_x, K] | None = None,
    ) -> NumPyGaussianBelief[int, K]:
        augmented = np.concatenate([states.array, inputs.array], axis=0)
        D_x = 2 * self.observation_dimension

        if covariances is None:
            # NOTE: No covariance means we are "certain" about the states.
            covariances = np.broadcast_to(  # type: ignore
                np.eye(D_x)[:, :, np.newaxis] * SMALL_UNCERTAINTY,
                (D_x, D_x, states.count),
            ).copy()

        assert shape_of(augmented, matches=(D_x, states.count), name="augmented state")
        assert shape_of(
            covariances, matches=(D_x, D_x, states.count), name="initial covariances"
        )

        return NumPyGaussianBelief(mean=augmented, covariance=covariances)

    @cached_property
    def state_transition_matrix(self) -> Array[Dims[D_x, D_x]]:
        D_o = self.observation_dimension

        # NOTE: State transition matrix for constant velocity model.
        return np.block(
            [
                [np.eye(D_o), self.time_step_size * np.eye(D_o)],
                [np.zeros((D_o, D_o)), np.eye(D_o)],
            ]
        )

    @cached_property
    def observation_matrix(self) -> Array[Dims[D_o, D_x]]:
        D_o = self.observation_dimension

        # NOTE: We have a velocity for each observed state.
        return np.hstack((np.eye(D_o), np.zeros((D_o, D_o))))


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorObstacleModel[D_o: int, D_x: int](
    ObstacleModel[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
        NumPyIntegratorObstacleCovariances[D_x, int],
        NumPyIntegratorObstacleStateSequences[int, D_x, int],
    ]
):
    """Propagates integrator dynamics forward with constant velocity."""

    model: NumPyIntegratorStateEstimationModel[D_o, D_x]
    process_noise_covariance: Array[Dims[D_x, D_x]]
    predictor: NumPyKalmanFilter

    @staticmethod
    def create[D_o_: int, D_x_: int](
        *,
        time_step_size: float,
        state_dimension: D_o_,
        process_noise_covariance: NumPyNoiseCovarianceDescription[D_x_] = 1e-3,
    ) -> "NumPyIntegratorObstacleModel[D_o_, D_x_]":
        """Creates a NumPy integrator obstacle model.

        Args:
            time_step_size: The time step size for integration.
            state_dimension: The dimension of the obstacle state (e.g., 2 for 2D position).
            process_noise_covariance: The process noise covariance, either as a
                full covariance array, a diagonal covariance vector, or a scalar
                variance representing isotropic noise.
        """
        return NumPyIntegratorObstacleModel(
            model=NumPyIntegratorStateEstimationModel.create(
                time_step_size=time_step_size, observation_dimension=state_dimension
            ),
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance,
                dimension=kf_state_dimension_for(observation_dimension=state_dimension),
            ),
            predictor=NumPyKalmanFilter.create(),
        )

    def forward[T: int, K: int](
        self,
        *,
        states: NumPyIntegratorObstacleStates[D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
        covariances: NumPyIntegratorObstacleCovariances[D_x, K] | None,
        horizon: T,
    ) -> NumPyIntegratorObstacleStateSequences[T, D_x, K]:
        beliefs = []
        last = self.model.initial_belief_from(
            states=states, inputs=inputs, covariances=covariances
        )

        for _ in range(horizon):
            beliefs.append(
                last := self.predictor.predict(
                    belief=last,
                    state_transition_matrix=self.model.state_transition_matrix,
                    process_noise_covariance=self.process_noise_covariance,
                )
            )

        return NumPyIntegratorObstacleStateSequences.create(beliefs)


@dataclass(frozen=True)
class NumPyFiniteDifferenceIntegratorStateEstimator(
    ObstacleStateEstimator[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
    ]
):
    time_step_size: float

    @staticmethod
    def create(
        *, time_step_size: float
    ) -> "NumPyFiniteDifferenceIntegratorStateEstimator":
        return NumPyFiniteDifferenceIntegratorStateEstimator(
            time_step_size=time_step_size
        )

    def estimate_from[D_o: int, K: int, T: int = int](
        self, history: NumPyIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates[D_o, K],
        NumPyIntegratorObstacleInputs[D_o, K],
        None,
    ]:
        """Estimates velocities from position history using finite differences.

        **Velocity** (requires T ≥ 2):
            $$v_t = (x_t - x_{t-1}) / (\\Delta t)$$
        """
        assert history.horizon > 0, (
            "History must contain at least one state for estimation."
        )

        filter_invalid = invalid_obstacle_filter_from(history, check_recent=2)
        velocities = self.estimate_velocities_from(history)

        return EstimatedObstacleStates(
            states=NumPyIntegratorObstacleStates.create(array=history.array[-1, :, :]),
            inputs=NumPyIntegratorObstacleInputs.create(
                array=filter_invalid(velocities)
            ),
            covariance=None,
        )

    def estimate_velocities_from[D_o: int, K: int, T: int = int](
        self, history: NumPyIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> Array[Dims[D_o, K]]:
        if history.horizon < 2:
            return cast(
                Array[Dims[D_o, K]], np.zeros((history.dimension, history.count))
            )

        return self._estimate_velocities_from(
            current=history.array[-1, :, :],
            previous=history.array[-2, :, :],
        )

    def _estimate_velocities_from[D_o: int, K: int](
        self,
        *,
        current: Array[Dims[D_o, K]],
        previous: Array[Dims[D_o, K]],
    ) -> Array[Dims[D_o, K]]:
        velocities = (current - previous) / self.time_step_size

        assert shape_of(
            velocities, matches=(current.shape[0], current.shape[1]), name="velocities"
        )

        return velocities


@dataclass(frozen=True)
class NumPyKfIntegratorStateEstimator[D_o: int, D_x: int](
    ObstacleStateEstimator[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
        NumPyIntegratorObstacleCovariances[D_x, int],
    ]
):
    """Kalman Filter state estimator for integrator model obstacles."""

    process_noise_covariance: Array[Dims[D_x, D_x]]
    observation_noise_covariance: Array[Dims[D_o, D_o]]
    model: NumPyIntegratorStateEstimationModel[D_o, D_x]
    estimator: NumPyKalmanFilter

    @staticmethod
    def create[D_o_: int, D_x_: int](
        *,
        time_step_size: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription[D_x_],
        observation_noise_covariance: NumPyNoiseCovarianceArrayDescription[D_o_],
        initial_state_covariance: Array[Dims[D_x_, D_x_]] | None = None,
        observation_dimension: D_o_ | None = None,
    ) -> "NumPyKfIntegratorStateEstimator[D_o_, D_x_]":
        """Creates an integrator state estimator based on the Kalman Filter with the
        specified noise covariances.

        Args:
            time_step_size: The time step size for the integrator.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a single scalar representing
                isotropic noise across all state dimensions.
            observation_noise_covariance: The observation noise covariance, either as
                a full matrix, a vector of diagonal entries, or a single scalar
                representing isotropic noise across all state dimensions.
            initial_state_covariance: The initial state covariance for the Kalman filter.
                If not provided, low uncertainty will be assumed for observed states and high
                uncertainty for unobserved velocities.
            observation_dimension: The observation dimension for the Kalman filter.
                Mandatory if noise/state covariances are not specified/specified as scalars.
        """
        observation_dimension = observation_dimension_from(
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            initial_state_covariance=initial_state_covariance,
            observation_dimension=observation_dimension,
        )

        return NumPyKfIntegratorStateEstimator(
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance,
                dimension=cast(D_x_, kf_state_dimension_for(observation_dimension)),
            ),
            observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=observation_dimension
            ),
            model=NumPyIntegratorStateEstimationModel.create(
                time_step_size=time_step_size,
                observation_dimension=observation_dimension,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=NumPyKalmanFilter.create(),
        )

    def estimate_from[K: int, T: int = int](
        self, history: NumPyIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates[D_o, K],
        NumPyIntegratorObstacleInputs[D_o, K],
        NumPyIntegratorObstacleCovariances[D_x, K],
    ]:
        """Estimate states and velocities using Kalman filtering."""
        assert history.horizon > 0, (
            "History must contain at least one state for estimation."
        )
        assert history.dimension == self.model.observation_dimension, (
            f"History dimension {history.dimension} does not match expected "
            f"observation dimension {self.model.observation_dimension}."
        )

        estimate = self.estimator.filter(
            observations=history.array,
            initial_state_covariance=self.model.initial_state_covariance,
            state_transition_matrix=self.model.state_transition_matrix,
            process_noise_covariance=self.process_noise_covariance,
            observation_noise_covariance=self.observation_noise_covariance,
            observation_matrix=self.model.observation_matrix,
        )

        return EstimatedObstacleStates(
            states=self.model.states_from(estimate),
            inputs=self.model.inputs_from(estimate),
            covariance=estimate.covariance,
        )


def validate_periodic_state_limits(state_limits: tuple[float, float] | None) -> None:
    assert state_limits is not None, (
        "Periodic boundaries require explicit state limits."
    )

    lower, upper = state_limits
    assert np.isfinite(lower) and np.isfinite(upper), (
        "Periodic boundaries must be finite."
    )

    assert upper > lower, (
        "Periodic boundaries require upper limit to be greater than lower limit."
    )


def simulate_with_state_limits[T: int, D_x: int, M: int](
    *,
    inputs: Array[Dims[T, D_x, M]],
    initial_state: Array[Dims[D_x]],
    time_step: float,
    state_limits: tuple[float, float],
) -> Array[Dims[T, D_x, M]]:
    deltas = inputs * time_step

    states = np.empty_like(deltas)
    states[0] = np.clip(initial_state[:, np.newaxis] + deltas[0], *state_limits)

    for t in range(1, deltas.shape[0]):
        states[t] = np.clip(states[t - 1] + deltas[t], *state_limits)

    return states


def wrap_periodic_state[D_x: int](
    *,
    state: Array[Dims[D_x]],
    state_limits: tuple[float, float],
) -> Array[Dims[D_x]]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + np.mod(state - lower, period)
    return np.where(state == upper, upper, wrapped)


def wrap_periodic_batch[T: int, D_x: int, M: int](
    *,
    states: Array[Dims[T, D_x, M]],
    state_limits: tuple[float, float],
) -> Array[Dims[T, D_x, M]]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + np.mod(states - lower, period)
    return np.where(states == upper, upper, wrapped)


def simulate[T: int, D_x: int, N: int](
    *,
    inputs: Array[Dims[T, D_x, N]],
    initial_states: Array[Dims[D_x, N]],
    time_step: float,
) -> Array[Dims[T, D_x, N]]:
    states = initial_states + np.cumsum(inputs * time_step, axis=0)
    return states
