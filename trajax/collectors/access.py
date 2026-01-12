from typing import Final, Sequence

from trajax.types import (
    SimulationDataAccessor,
    StateSequence,
    Control,
    Risk,
    ObstacleStates,
    StateTrajectories,
)


class access:
    states: Final = SimulationDataAccessor.create(StateSequence, key="states")
    controls: Final = SimulationDataAccessor.create(Sequence[Control], key="controls")
    risks: Final = SimulationDataAccessor.create(Sequence[Risk], key="risks")
    trajectories: Final = SimulationDataAccessor.create(
        Sequence[StateTrajectories], key="trajectories"
    )
    obstacle_states: Final = SimulationDataAccessor.create(
        ObstacleStates, key="obstacle_states"
    )
    obstacle_forecasts: Final = SimulationDataAccessor.create(
        Sequence[ObstacleStates], key="obstacle_forecasts"
    )
