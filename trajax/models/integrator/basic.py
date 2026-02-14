from typing import Final, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    HasShape,
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

from numtypes import Array, Dims, shape_of

import numpy as np


NO_LIMITS: Final = (float("-inf"), float("inf"))


@dataclass(frozen=True)
class NumPyIntegratorObstacleStates[D_o: int, K: int]:
    """Arbitrary obstacle states with no semantic meaning attached."""

    _array: Array[Dims[D_o, K]]
    _covariance: Array[Dims[D_o, D_o, K]] | None

    @staticmethod
    def create(
        *,
        array: Array[Dims[D_o, K]],
        covariance: Array[Dims[D_o, D_o, K]] | None = None,
    ) -> "NumPyIntegratorObstacleStates[D_o, K]":
        return NumPyIntegratorObstacleStates(_array=array, _covariance=covariance)

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self.array

    def covariance(self) -> Array[Dims[D_o, D_o, K]] | None:
        return self._covariance

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
class NumPyIntegratorObstacleStateSequences[T: int, D_o: int, K: int]:
    """Time-indexed obstacle state sequences for integrator model obstacles."""

    array: Array[Dims[T, D_o, K]]

    def __array__(self, dtype: None | type = None) -> Array[Dims[T, D_o, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


@dataclass(frozen=True)
class NumPyIntegratorObstacleInputs[D_o: int, K: int]:
    _array: Array[Dims[D_o, K]]
    _covariance: Array[Dims[D_o, D_o, K]] | None

    @staticmethod
    def create(
        *,
        array: Array[Dims[D_o, K]],
        covariance: Array[Dims[D_o, D_o, K]] | None = None,
    ) -> "NumPyIntegratorObstacleInputs[D_o, K]":
        return NumPyIntegratorObstacleInputs(_array=array, _covariance=covariance)

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self.array

    def zeroed(self, *, at: tuple[int, ...]) -> "NumPyIntegratorObstacleInputs[D_o, K]":
        """Returns new obstacle inputs with inputs at specified state dimensions zeroed out."""

        zeroed_array = self.array.copy()
        zeroed_array[at, :] = 0.0

        return NumPyIntegratorObstacleInputs.create(
            array=zeroed_array, covariance=self._covariance
        )

    def covariance(self) -> Array[Dims[D_o, D_o, K]] | None:
        return self._covariance

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
class NumPyIntegratorObstacleControlInputSequences[T: int, D_o: int, K: int]:
    array: Array[Dims[T, D_o, K]]

    def __array__(self, dtype: None | type = None) -> Array[Dims[T, D_o, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


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


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorObstacleModel(
    ObstacleModel[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
        NumPyIntegratorObstacleStateSequences,
    ]
):
    """Propagates integrator dynamics forward with constant velocity."""

    time_step: float

    @staticmethod
    def create(*, time_step_size: float) -> "NumPyIntegratorObstacleModel":
        """Creates a NumPy integrator obstacle model.

        See `NumPyIntegratorModel.create` for details on the integrator dynamics.
        """
        return NumPyIntegratorObstacleModel(time_step=time_step_size)

    def forward[T: int, D_o: int, K: int](
        self,
        *,
        current: NumPyIntegratorObstacleStates[D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
        horizon: T,
    ) -> NumPyIntegratorObstacleStateSequences[T, D_o, K]:
        input_sequences = self._input_to_maintain(inputs, horizon=horizon)

        result = simulate(
            inputs=input_sequences.array,
            initial_states=current.array,
            time_step=self.time_step,
        )

        return NumPyIntegratorObstacleStateSequences(result)

    def state_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: NumPyIntegratorObstacleStateSequences[T, D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
    ) -> Array[Dims[T, D_o, D_o, K]]:
        raise NotImplementedError(
            "State Jacobian is not implemented for NumPyIntegratorObstacleModel."
        )

    def input_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: NumPyIntegratorObstacleStateSequences[T, D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
    ) -> Array[Dims[T, D_o, D_o, K]]:
        raise NotImplementedError(
            "Input Jacobian is not implemented for NumPyIntegratorObstacleModel."
        )

    def _input_to_maintain[T: int, D_o: int, K: int](
        self,
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
        *,
        horizon: T,
    ) -> NumPyIntegratorObstacleControlInputSequences[T, D_o, K]:
        return NumPyIntegratorObstacleControlInputSequences(
            np.tile(inputs.array[np.newaxis, :, :], (horizon, 1, 1))
        )


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
        assert history.horizon > 0, (
            "History must contain at least one state for estimation."
        )

        velocities = self.estimate_velocities_from(history)

        return EstimatedObstacleStates(
            states=NumPyIntegratorObstacleStates.create(array=history.array[-1, :, :]),
            inputs=NumPyIntegratorObstacleInputs.create(array=velocities),
            covariance=None,
        )

    def estimate_velocities_from[D_o: int, K: int, T: int = int](
        self, history: NumPyIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> Array[Dims[D_o, K]]:
        """Estimates velocities from position history using finite differences.

        $$v_t = (x_t - x_{t-1}) / (\\Delta t)$$
        """
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
    ]
):
    """Kalman Filter state estimator for integrator model obstacles."""

    time_step_size: float
    process_noise_covariance: Array[Dims[D_x, D_x]]
    observation_noise_covariance: Array[Dims[D_o, D_o]]
    observation_dimension: D_o
    estimator: NumPyKalmanFilter

    @staticmethod
    def create[D_o_: int, D_x_: int](
        *,
        time_step_size: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription[D_x_],
        observation_noise_covariance: NumPyNoiseCovarianceArrayDescription[D_o_],
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
            observation_dimension: The observation dimension for the Kalman filter.
                Mandatory if both noise covariances are specified as scalars.
        """
        observation_dimension = observation_dimension_from(
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            observation_dimension=observation_dimension,
        )

        return NumPyKfIntegratorStateEstimator(
            time_step_size=time_step_size,
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance,
                dimension=cast(D_x_, kf_state_dimension_for(observation_dimension)),
            ),
            observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=observation_dimension
            ),
            observation_dimension=observation_dimension,
            estimator=NumPyKalmanFilter.create(),
        )

    def estimate_from[K: int, T: int = int](
        self, history: NumPyIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates[D_o, K],
        NumPyIntegratorObstacleInputs[D_o, K],
        Array[Dims[D_x, D_x, K]],
    ]:
        """Estimate states and velocities using Kalman filtering."""
        assert history.horizon > 0, (
            "History must contain at least one state for estimation."
        )
        assert history.dimension == self.observation_dimension, (
            f"History dimension {history.dimension} does not match expected "
            f"observation dimension {self.observation_dimension}."
        )

        estimate = self.estimator.filter(
            observations=history.array,
            initial_state_covariance=self.initial_state_covariance,
            state_transition_matrix=self.state_transition_matrix,
            process_noise_covariance=self.process_noise_covariance,
            observation_noise_covariance=self.observation_noise_covariance,
            observation_matrix=self.observation_matrix,
        )

        return EstimatedObstacleStates(
            states=self.states_from(estimate),
            inputs=self.inputs_from(estimate),
            covariance=estimate.covariance,
        )

    def states_from[K: int](
        self, belief: NumPyGaussianBelief[D_x, K]
    ) -> NumPyIntegratorObstacleStates[D_o, K]:
        D_o = self.observation_dimension

        return NumPyIntegratorObstacleStates.create(
            array=belief.mean[:D_o, :], covariance=belief.covariance[:D_o, :D_o, :]
        )

    def inputs_from[K: int](
        self, belief: NumPyGaussianBelief[D_x, K]
    ) -> NumPyIntegratorObstacleInputs[D_o, K]:
        D_o = self.observation_dimension

        return NumPyIntegratorObstacleInputs.create(
            array=belief.mean[D_o:, :], covariance=belief.covariance[D_o:, D_o:, :]
        )

    @cached_property
    def initial_state_covariance(self) -> Array[Dims[D_x, D_x]]:
        D_o = self.observation_dimension

        # NOTE: We are sure of the observed states, unsure of the velocities.
        return np.diag(np.concatenate((np.full(D_o, 1e-4), np.full(D_o, 1e3))))

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


def observation_dimension_from[D_o: int, D_x: int](
    *,
    process_noise_covariance: NumPyNoiseCovarianceDescription[D_x],
    observation_noise_covariance: NumPyNoiseCovarianceDescription[D_o],
    observation_dimension: D_o | None,
) -> D_o:
    if observation_dimension is not None:
        return observation_dimension

    if isinstance(observation_noise_covariance, HasShape):
        return cast(D_o, observation_noise_covariance.shape[0])

    if isinstance(process_noise_covariance, HasShape):
        return cast(D_o, observation_dimension_for(process_noise_covariance.shape[0]))

    assert False, (
        "Observation dimension must be specified if noise covariances are not provided as arrays."
    )


def kf_state_dimension_for(observation_dimension: int) -> int:
    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return 2 * observation_dimension


def observation_dimension_for(kf_state_dimension: int) -> int:
    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return kf_state_dimension // 2
