from dataclasses import dataclass

from faran.types.mppi import StateSequence


@dataclass(frozen=True)
class StateTrajectories[TrajectoriesT = StateSequence]:
    optimal: TrajectoriesT
    nominal: TrajectoriesT
