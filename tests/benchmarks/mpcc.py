from dataclasses import dataclass

from faran import (
    State,
    ControlInputSequence,
    Mppi,
    ObstacleSimulator,
    ObstacleStateObserver,
)


@dataclass(frozen=True)
class MpccConfiguration[StateT: State, InputT: ControlInputSequence]:
    planner: Mppi[StateT, InputT]
    initial_state: StateT
    nominal_input: InputT

    obstacle_simulator: ObstacleSimulator | None = None
    obstacle_state_observer: ObstacleStateObserver | None = None

    def __repr__(self) -> str:
        return f"MpccConfiguration(planner={self.planner.__class__.__name__})"


def accumulate_obstacle_states(
    configuration: MpccConfiguration, steps: int = 5
) -> None:
    simulator = configuration.obstacle_simulator
    observer = configuration.obstacle_state_observer

    assert simulator is not None, "Obstacle simulator is required to accumulate states."
    assert observer is not None, (
        "Obstacle state observer is required to accumulate states."
    )

    # NOTE: To collect sufficient obstacle state history, we simulate a few steps
    # for the obstacles.
    for _ in range(steps):
        observer.observe(simulator.step())
