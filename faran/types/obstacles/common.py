from typing import Protocol, Self


class ObstacleSimulator[ObstacleStatesForTimeStepT](Protocol):
    def step(self) -> ObstacleStatesForTimeStepT:
        """Advances the obstacle simulation by one time step and returns the new states."""
        ...

    def with_time_step_size(self, time_step_size: float) -> Self:
        """Returns a new simulator with the specified time step size."""
        ...

    @property
    def obstacle_count(self) -> int:
        """Returns the number of obstacles being simulated."""
        ...
