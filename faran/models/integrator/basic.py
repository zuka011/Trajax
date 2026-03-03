from typing import Final, Sequence
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
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
    NumPyGaussianBelief,
    NumPyNoiseCovarianceArrayDescription,
    NumPyNoiseCovarianceDescription,
    NumPyNoiseModelProvider,
)
from faran.states import (
    NumPySimpleState as SimpleState,
    NumPySimpleStateSequence as SimpleStateSequence,
    NumPySimpleStateBatch as SimpleStateBatch,
    NumPySimpleControlInputBatch as SimpleControlInputBatch,
)
from faran.filters import NumPyKalmanFilter, numpy_kalman_filter
from faran.models.common import SMALL_UNCERTAINTY, LARGE_UNCERTAINTY
from faran.models.basic import invalid_obstacle_filter_from
from faran.models.integrator.common import (
    observation_dimension_from,
    kf_state_dimension_for,
)


from jaxtyping import Float

import numpy as np


NO_LIMITS: Final = (float("-inf"), float("inf"))

type NumPyIntegratorObstacleCovariances = Float[Array, "D_x D_x K"]


@jaxtyped
@dataclass(frozen=True)
class NumPyIntegratorObstacleStates:
    """Arbitrary obstacle states with no semantic meaning attached."""

    _array: Float[Array, "D_o K"]

    @staticmethod
    def create(*, array: Float[Array, "D_o K"]) -> "NumPyIntegratorObstacleStates":
        return NumPyIntegratorObstacleStates(_array=array)

    def __array__(self, dtype: None | type = None) -> Float[Array, "D_o K"]:
        return self.array

    @property
    def dimension(self) -> int:
        return self.array.shape[0]

    @property
    def count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "D_o K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyIntegratorObstacleStateSequences:
    """Time-indexed obstacle state sequences for integrator model obstacles."""

    _array: Float[Array, "T D_x K"]
    _covariance: Float[Array, "T D_x D_x K"]

    @staticmethod
    def create(
        predictions: Sequence[NumPyGaussianBelief],
    ) -> "NumPyIntegratorObstacleStateSequences":
        assert len(predictions) > 0, "Predictions sequence must not be empty."

        return NumPyIntegratorObstacleStateSequences(
            _array=np.stack([belief.mean for belief in predictions], axis=0),
            _covariance=np.stack([belief.covariance for belief in predictions], axis=0),
        )

    def __array__(self, dtype: None | type = None) -> Float[Array, "T D_x K"]:
        return self.array

    def covariance(self) -> Float[Array, "T D_x D_x K"]:
        return self._covariance

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T D_x K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyIntegratorObstacleInputs:
    _array: Float[Array, "D_o K"]

    @staticmethod
    def create(*, array: Float[Array, "D_o K"]) -> "NumPyIntegratorObstacleInputs":
        return NumPyIntegratorObstacleInputs(_array=array)

    def __array__(self, dtype: None | type = None) -> Float[Array, "D_o K"]:
        return self.array

    def zeroed(self, *, at: tuple[int, ...]) -> "NumPyIntegratorObstacleInputs":
        """Returns new obstacle inputs with inputs at specified state dimensions zeroed out."""

        zeroed_array = self.array.copy()
        zeroed_array[at, :] = 0.0

        return NumPyIntegratorObstacleInputs.create(array=zeroed_array)

    @property
    def dimension(self) -> int:
        return self.array.shape[0]

    @property
    def count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "D_o K"]:
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

    def simulate(
        self,
        inputs: NumPyIntegratorControlInputBatch,
        initial_state: NumPyIntegratorState,
    ) -> SimpleStateBatch:
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

    def step(
        self,
        inputs: NumPyIntegratorControlInputSequence,
        state: NumPyIntegratorState,
    ) -> SimpleState:
        clipped_control = np.clip(inputs.array[0], *self.velocity_limits)
        unbounded = state.array + clipped_control * self.time_step_size
        new_state = (
            wrap_periodic_state(state=unbounded, state_limits=self.state_limits)
            if self.periodic
            else np.clip(unbounded, *self.state_limits)
        )

        return SimpleState(new_state)

    def forward(
        self,
        inputs: NumPyIntegratorControlInputSequence,
        state: NumPyIntegratorState,
    ) -> SimpleStateSequence:
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
class NumPyIntegratorStateEstimationModel:
    """Single integrator model used for obstacle state estimation."""

    time_step_size: float
    observation_dimension: int
    initial_state_covariance: Float[Array, "D_x D_x"]

    @staticmethod
    def create(
        *,
        time_step_size: float,
        observation_dimension: int,
        initial_state_covariance: Float[Array, "D_x D_x"] | None = None,
    ) -> "NumPyIntegratorStateEstimationModel":
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
    def default_initial_state_covariance_for(
        observation_dimension: int,
    ) -> Float[Array, "D_x D_x"]:
        D_o_ = observation_dimension

        # NOTE: We are sure of the observed states, unsure of the velocities.
        return np.diag(
            np.concatenate(
                (np.full(D_o_, SMALL_UNCERTAINTY), np.full(D_o_, LARGE_UNCERTAINTY))
            )
        )

    def __call__(self, augmented_state: Float[Array, "D_x K"]) -> Float[Array, "D_x K"]:
        return self.state_transition_matrix @ augmented_state

    def states_from(self, belief: NumPyGaussianBelief) -> NumPyIntegratorObstacleStates:
        D_o = self.observation_dimension
        return NumPyIntegratorObstacleStates.create(array=belief.mean[:D_o, :])

    def inputs_from(self, belief: NumPyGaussianBelief) -> NumPyIntegratorObstacleInputs:
        D_o = self.observation_dimension
        return NumPyIntegratorObstacleInputs.create(array=belief.mean[D_o:, :])

    def initial_belief_from(
        self,
        *,
        states: NumPyIntegratorObstacleStates,
        inputs: NumPyIntegratorObstacleInputs,
        covariances: NumPyIntegratorObstacleCovariances | None = None,
    ) -> NumPyGaussianBelief:
        augmented = np.concatenate([states.array, inputs.array], axis=0)
        D_x = 2 * self.observation_dimension

        if covariances is None:
            # NOTE: No covariance means we are "certain" about the states.
            covariances = np.broadcast_to(  # type: ignore
                np.eye(D_x)[:, :, np.newaxis] * SMALL_UNCERTAINTY,
                (D_x, D_x, states.count),
            ).copy()

        return NumPyGaussianBelief(mean=augmented, covariance=covariances)

    @cached_property
    def state_transition_matrix(self) -> Float[Array, "D_x D_x"]:
        D_o = self.observation_dimension

        # NOTE: State transition matrix for constant velocity model.
        return np.block(
            [
                [np.eye(D_o), self.time_step_size * np.eye(D_o)],
                [np.zeros((D_o, D_o)), np.eye(D_o)],
            ]
        )

    @cached_property
    def observation_matrix(self) -> Float[Array, "D_o D_x"]:
        D_o = self.observation_dimension

        # NOTE: We have a velocity for each observed state.
        return np.hstack((np.eye(D_o), np.zeros((D_o, D_o))))


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorObstacleModel(
    ObstacleModel[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
        NumPyIntegratorObstacleCovariances,
        NumPyIntegratorObstacleStateSequences,
    ]
):
    """Propagates integrator dynamics forward with constant velocity."""

    model: NumPyIntegratorStateEstimationModel
    process_noise_covariance: Float[Array, "D_x D_x"]
    predictor: NumPyKalmanFilter

    @staticmethod
    def create(
        *,
        time_step_size: float,
        state_dimension: int,
        process_noise_covariance: NumPyNoiseCovarianceDescription = 1e-3,
    ) -> "NumPyIntegratorObstacleModel":
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

    def forward(
        self,
        *,
        states: NumPyIntegratorObstacleStates,
        inputs: NumPyIntegratorObstacleInputs,
        covariances: NumPyIntegratorObstacleCovariances | None,
        horizon: int,
    ) -> NumPyIntegratorObstacleStateSequences:
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

    def estimate_from(
        self, history: NumPyIntegratorObstacleStatesHistory
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
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

    def estimate_velocities_from(
        self, history: NumPyIntegratorObstacleStatesHistory
    ) -> Float[Array, "D_o K"]:
        if history.horizon < 2:
            return np.zeros((history.dimension, history.count))

        return self._estimate_velocities_from(
            current=history.array[-1, :, :],
            previous=history.array[-2, :, :],
        )

    def _estimate_velocities_from(
        self, *, current: Float[Array, "D_o K"], previous: Float[Array, "D_o K"]
    ) -> Float[Array, "D_o K"]:
        velocities = (current - previous) / self.time_step_size

        return velocities


@dataclass(frozen=True)
class NumPyKfIntegratorStateEstimator(
    ObstacleStateEstimator[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
        NumPyIntegratorObstacleCovariances,
    ]
):
    """Kalman Filter state estimator for integrator model obstacles."""

    process_noise_covariance: Float[Array, "D_x D_x"]
    observation_noise_covariance: Float[Array, "D_o D_o"]
    model: NumPyIntegratorStateEstimationModel
    estimator: NumPyKalmanFilter

    @staticmethod
    def create(
        *,
        time_step_size: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription,
        observation_noise_covariance: NumPyNoiseCovarianceArrayDescription,
        initial_state_covariance: Float[Array, "D_x D_x"] | None = None,
        observation_dimension: int | None = None,
        noise_model: NumPyNoiseModelProvider | None = None,
    ) -> "NumPyKfIntegratorStateEstimator":
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
                dimension=kf_state_dimension_for(observation_dimension),
            ),
            observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=observation_dimension
            ),
            model=NumPyIntegratorStateEstimationModel.create(
                time_step_size=time_step_size,
                observation_dimension=observation_dimension,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=NumPyKalmanFilter.create(noise_model=noise_model),
        )

    def estimate_from(
        self, history: NumPyIntegratorObstacleStatesHistory
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
        NumPyIntegratorObstacleCovariances,
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


def simulate_with_state_limits(
    *,
    inputs: Float[Array, "T D_x M"],
    initial_state: Float[Array, " D_x"],
    time_step: float,
    state_limits: tuple[float, float],
) -> Float[Array, "T D_x M"]:
    deltas = inputs * time_step

    states = np.empty_like(deltas)
    states[0] = np.clip(initial_state[:, np.newaxis] + deltas[0], *state_limits)

    for t in range(1, deltas.shape[0]):
        states[t] = np.clip(states[t - 1] + deltas[t], *state_limits)

    return states


def wrap_periodic_state(
    *,
    state: Float[Array, " D_x"],
    state_limits: tuple[float, float],
) -> Float[Array, " D_x"]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + np.mod(state - lower, period)
    return np.where(state == upper, upper, wrapped)


def wrap_periodic_batch(
    *,
    states: Float[Array, "T D_x M"],
    state_limits: tuple[float, float],
) -> Float[Array, "T D_x M"]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + np.mod(states - lower, period)
    return np.where(states == upper, upper, wrapped)


def simulate(
    *,
    inputs: Float[Array, "T D_x N"],
    initial_states: Float[Array, "D_x N"],
    time_step: float,
) -> Float[Array, "T D_x N"]:
    states = initial_states + np.cumsum(inputs * time_step, axis=0)
    return states
