from dataclasses import dataclass

from trajax.mppi import StateBatch
from trajax.costs.collision import (
    ObstacleStates,
    ObstacleStateSampler,
    SampledObstacleStates,
)


@dataclass(frozen=True)
class StateTrajectories[StateT: StateBatch]:
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
    StateT: ObstacleStates,
    SampleT: SampledObstacleStates,
]:
    obstacle_states: StateT
    sampler: ObstacleStateSampler[StateT, SampleT]

    def sample(self, count: int) -> SampleT:
        return self.sampler(self.obstacle_states, count=count)
