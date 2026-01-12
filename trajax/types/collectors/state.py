from dataclasses import dataclass

from trajax.types.mppi import StateSequence


@dataclass(frozen=True)
class StateTrajectories[TrajectoriesT = StateSequence]:
    optimal: TrajectoriesT
    nominal: TrajectoriesT
