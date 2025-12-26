from typing import Protocol
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi import (
    StateBatch,
    CostFunction,
    NumPyStateBatch,
    NumPyControlInputBatch,
)
from trajax.states import NumPySimpleCosts
from trajax.costs.risk import RiskMetric, NoMetric
from trajax.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    Distance,
    D_o,
)


from numtypes import Array, Dims, D
from riskit import risk, sampler

import riskit
import numpy as np

type SampleCostFunction[StateT, SampleT] = riskit.NumPyBatchCostFunction[
    StateT, SampleT
]


class NumPySampledObstacleStates[T: int, K: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        """Returns the sampled states of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K, N]]:
        """Returns the sampled x positions of obstacles over time."""
        ...

    def y(self) -> Array[Dims[T, K, N]]:
        """Returns the sampled y positions of obstacles over time."""
        ...

    def heading(self) -> Array[Dims[T, K, N]]:
        """Returns the sampled headings of obstacles over time."""
        ...


class NumPyObstacleStates[T: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the means states of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K]]:
        """Returns the mean x positions of obstacles over time."""
        ...

    def y(self) -> Array[Dims[T, K]]:
        """Returns the mean y positions of obstacles over time."""
        ...

    def heading(self) -> Array[Dims[T, K]]:
        """Returns the mean headings of obstacles over time."""
        ...

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        """Returns the covariance matrices for the obstacle states x, y, heading over time, if
        one exists."""
        ...

    def sampled[N: int](
        self,
        *,
        x: Array[Dims[T, K, N]],
        y: Array[Dims[T, K, N]],
        heading: Array[Dims[T, K, N]],
    ) -> NumPySampledObstacleStates[T, K, N]:
        """Returns sampled states of obstacles over time."""
        ...

    def single(self) -> NumPySampledObstacleStates[T, K, D[1]]:
        """Returns the single (mean) sample of the obstacle states."""
        ...


class NumPyDistanceExtractor[
    StateT: NumPyStateBatch,
    SampleT: NumPySampledObstacleStates,
    DistanceT: "NumPyDistance",
](Protocol):
    def __call__(self, *, states: StateT, obstacle_states: SampleT) -> DistanceT:
        """Extracts minimum distances to obstacles in the environment from a batch of states."""
        ...


class NumPyObstacleStateProvider[ObstacleStateT: NumPyObstacleStates](
    ObstacleStateProvider[ObstacleStateT], Protocol
): ...


class NumPyObstacleStateSampler[
    StateT: NumPyObstacleStates,
    SampleT: NumPySampledObstacleStates,
](ObstacleStateSampler[StateT, SampleT], Protocol): ...


@dataclass(frozen=True)
class NumPyDistance[T: int, V: int, M: int, N: int](Distance[T, V, M, N]):
    array: Array[Dims[T, V, M, N]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M, N]]:
        return self.array


@dataclass(kw_only=True, frozen=True)
class NumPyCollisionCost[
    StateT: NumPyStateBatch,
    ObstacleStatesT: NumPyObstacleStates,
    SampledObstacleStatesT: NumPySampledObstacleStates,
    DistanceT: NumPyDistance,
    V: int,
](CostFunction[NumPyControlInputBatch[int, int, int], StateT, NumPySimpleCosts]):
    obstacle_states: NumPyObstacleStateProvider[ObstacleStatesT]
    sampler: NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT]
    distance: NumPyDistanceExtractor[StateT, SampledObstacleStatesT, DistanceT]
    distance_threshold: Array[Dims[V]]
    weight: float
    metric: RiskMetric

    @staticmethod
    def create[
        S: StateBatch,
        OS: NumPyObstacleStates,
        SOS: NumPySampledObstacleStates,
        D: NumPyDistance,
        V_: int,
    ](
        *,
        obstacle_states: NumPyObstacleStateProvider[OS],
        sampler: NumPyObstacleStateSampler[OS, SOS],
        distance: NumPyDistanceExtractor[S, SOS, D],
        distance_threshold: Array[Dims[V_]],
        weight: float,
        metric: RiskMetric = NoMetric(),
    ) -> "NumPyCollisionCost[S, OS, SOS, D, V_]":
        return NumPyCollisionCost(
            obstacle_states=obstacle_states,
            sampler=sampler,
            distance=distance,
            distance_threshold=distance_threshold,
            weight=weight,
            metric=metric,
        )

    def __call__[T: int, M: int](
        self, *, inputs: NumPyControlInputBatch[T, int, M], states: StateT
    ) -> NumPySimpleCosts[T, M]:
        metric = risk.expected_value_of(self.cost_function()).sampled_with(
            sampler.monte_carlo(self.metric.sample_count)
        )

        return NumPySimpleCosts(
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
        ) -> riskit.NumPyCosts:
            cost = self.distance_threshold[
                np.newaxis, :, np.newaxis, np.newaxis
            ] - np.asarray(
                self.distance(states=trajectories, obstacle_states=uncertainties)
            )

            return self.weight * np.clip(cost, 0, None).sum(axis=1)

        return J


@dataclass(frozen=True)
class StateTrajectories[StateT: NumPyStateBatch]:
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
    StateT: NumPyObstacleStates,
    SampleT: NumPySampledObstacleStates,
]:
    obstacle_states: StateT
    sampler: NumPyObstacleStateSampler[StateT, SampleT]

    def sample(self, count: int) -> SampleT:
        return self.sampler(self.obstacle_states, count=count)
