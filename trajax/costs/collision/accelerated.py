from typing import Protocol
from dataclasses import dataclass

from trajax.type import jaxtyped, DataType
from trajax.mppi import (
    CostFunction,
    JaxControlInputBatch,
    JaxStateBatch,
)
from trajax.states import JaxSimpleCosts
from trajax.costs.risk import RiskMetric, NoMetric
from trajax.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    Distance,
    D_o,
)

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D
from riskit import risk, sampler

import riskit
import numpy as np
import jax
import jax.numpy as jnp


type SampleCostFunction[StateT, SampleT] = riskit.JaxBatchCostFunction[StateT, SampleT]


class JaxSampledObstacleStates[T: int, K: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        """Returns the sampled states of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K, N]]:
        """Returns the x positions of obstacles over time and samples as a NumPy array."""
        ...

    def y(self) -> Array[Dims[T, K, N]]:
        """Returns the y positions of obstacles over time and samples as a NumPy array."""
        ...

    def heading(self) -> Array[Dims[T, K, N]]:
        """Returns the headings of obstacles over time and samples as a NumPy array."""
        ...

    @property
    def x_array(self) -> Float[JaxArray, "T K N"]:
        """Returns the x positions of obstacles over time and samples."""
        ...

    @property
    def y_array(self) -> Float[JaxArray, "T K N"]:
        """Returns the y positions of obstacles over time and samples."""
        ...

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        """Returns the headings of obstacles over time and samples."""
        ...

    @property
    def sample_count(self) -> N:
        """Returns the number of samples per obstacle."""
        ...


class JaxObstacleStates[T: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the states of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K]]:
        """Returns the x positions of obstacles over time as a NumPy array."""
        ...

    def y(self) -> Array[Dims[T, K]]:
        """Returns the y positions of obstacles over time as a NumPy array."""
        ...

    def heading(self) -> Array[Dims[T, K]]:
        """Returns the headings of obstacles over time as a NumPy array."""
        ...

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        """Returns the covariance matrices of obstacles over time as a NumPy array."""
        ...

    def sampled[N: int = int](
        *,
        x: Float[JaxArray, "T K N"],
        y: Float[JaxArray, "T K N"],
        heading: Float[JaxArray, "T K N"],
        sample_count: N | None = None,
    ) -> JaxSampledObstacleStates[T, K, N]:
        """Returns sampled states of obstacles over time."""
        ...

    def single(self) -> JaxSampledObstacleStates[T, K, D[1]]:
        """Returns the single (mean) sample of the obstacle states."""
        ...

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        """Returns the x positions of obstacles over time."""
        ...

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        """Returns the y positions of obstacles over time."""
        ...

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        """Returns the headings of obstacles over time."""
        ...

    @property
    def covariance_array(self) -> Float[JaxArray, "T D_o D_o K"] | None:
        """Returns the covariance matrices of obstacles over time."""
        ...


class JaxDistanceExtractor[
    StateT: JaxStateBatch,
    SampleT: JaxSampledObstacleStates,
    DistanceT: "JaxDistance",
](Protocol):
    def __call__(self, *, states: StateT, obstacle_states: SampleT) -> DistanceT:
        """Extracts minimum distances to obstacles in the environment from a batch of states."""
        ...


class JaxObstacleStateProvider[ObstacleStateT: JaxObstacleStates](
    ObstacleStateProvider[ObstacleStateT], Protocol
): ...


class JaxObstacleStateSampler[
    StateT: JaxObstacleStates,
    SampleT: JaxSampledObstacleStates,
](ObstacleStateSampler[StateT, SampleT], Protocol): ...


@jaxtyped
@dataclass(frozen=True)
class JaxDistance[T: int, V: int, M: int, N: int](Distance[T, V, M, N]):
    _array: Float[JaxArray, "T V M N"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M, N]]:
        return np.asarray(self.array)

    @property
    def array(self) -> Float[JaxArray, "T V M N"]:
        return self._array


@dataclass(kw_only=True, frozen=True)
class JaxCollisionCost[
    StateT: JaxStateBatch,
    ObstacleStatesT: JaxObstacleStates,
    SampledObstacleStatesT: JaxSampledObstacleStates,
    DistanceT: JaxDistance,
    V: int,
](CostFunction[JaxControlInputBatch[int, int, int], StateT, JaxSimpleCosts]):
    obstacle_states: JaxObstacleStateProvider[ObstacleStatesT]
    sampler: JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT]
    distance: JaxDistanceExtractor[StateT, SampledObstacleStatesT, DistanceT]
    distance_threshold: Float[JaxArray, "V"]
    weight: float
    metric: RiskMetric

    @staticmethod
    def create[
        S: JaxStateBatch,
        OS: JaxObstacleStates,
        SOS: JaxSampledObstacleStates,
        D: JaxDistance,
        V_: int,
    ](
        *,
        obstacle_states: JaxObstacleStateProvider[OS],
        sampler: JaxObstacleStateSampler[OS, SOS],
        distance: JaxDistanceExtractor[S, SOS, D],
        distance_threshold: Array[Dims[V_]],
        weight: float,
        metric: RiskMetric = NoMetric(),
    ) -> "JaxCollisionCost[S, OS, SOS, D, V_]":
        return JaxCollisionCost(
            obstacle_states=obstacle_states,
            sampler=sampler,
            distance=distance,
            distance_threshold=jnp.asarray(distance_threshold),
            weight=weight,
            metric=metric,
        )

    def __call__[T: int, M: int](
        self, *, inputs: JaxControlInputBatch[T, int, M], states: StateT
    ) -> JaxSimpleCosts[T, M]:
        metric = risk.expected_value_of(self.cost_function()).sampled_with(
            sampler.monte_carlo(self.metric.sample_count)
        )

        return JaxSimpleCosts(
            metric.compute(
                trajectories=StateTrajectories(states),
                uncertainties=ObstacleStateUncertainties(
                    obstacle_states=self.obstacle_states(), sampler=self.sampler
                ),
            )
        )

    def cost_function(self) -> SampleCostFunction[StateT, SampledObstacleStatesT]:
        def J(
            *, trajectories: StateT, uncertainties: SampledObstacleStatesT
        ) -> riskit.JaxCosts:
            return collision_cost(
                distance=self.distance(
                    states=trajectories, obstacle_states=uncertainties
                ).array,
                distance_threshold=self.distance_threshold,
                weight=self.weight,
            )

        return J


@dataclass(frozen=True)
class StateTrajectories[StateT: JaxStateBatch]:
    states: StateT

    def get(self) -> StateT:
        return self.states

    @property
    def time_steps(self) -> int:
        return self.states.horizon

    @property
    def trajectory_count(self) -> int:
        return self.states.rollout_count


@dataclass(frozen=True)
class ObstacleStateUncertainties[
    StateT: JaxObstacleStates,
    SampleT: JaxSampledObstacleStates,
]:
    obstacle_states: StateT
    sampler: JaxObstacleStateSampler[StateT, SampleT]

    def sample(self, count: int) -> SampleT:
        return self.sampler(self.obstacle_states, count=count)


@jax.jit
@jaxtyped
def collision_cost(
    *,
    distance: Float[JaxArray, "T V M N"],
    distance_threshold: Float[JaxArray, "V"],
    weight: Scalar,
) -> Float[JaxArray, "T M N"]:
    cost = distance_threshold[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] - distance
    return weight * jnp.clip(cost, 0, None).sum(axis=1)
