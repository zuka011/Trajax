from typing import Final, cast
from dataclasses import dataclass

from trajax.types import (
    ObstacleIdAssignment,
    ObstacleStatesForTimeStep,
    ObstacleStatesHistory,
    NumPyObstaclePositionsForTimeStep,
    NumPyObstaclePositions,
    NumPyObstaclePositionExtractor,
)
from trajax.obstacles.history import NumPyObstacleIds

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from numtypes import Array, IndexArray, BoolArray, Dims

import numpy as np


type PositionsExtractor[StatesT, HistoryT] = NumPyObstaclePositionExtractor[
    StatesT, HistoryT, NumPyObstaclePositionsForTimeStep, NumPyObstaclePositions
]


@dataclass(kw_only=True)
class NumPyHungarianObstacleIdAssignment[
    StatesT: ObstacleStatesForTimeStep,
    HistoryT: ObstacleStatesHistory,
](ObstacleIdAssignment[StatesT, NumPyObstacleIds, HistoryT]):
    positions: Final[PositionsExtractor[StatesT, HistoryT]]
    cutoff: Final[float]
    next_id: int

    @staticmethod
    def create[S: ObstacleStatesForTimeStep, H: ObstacleStatesHistory](
        *,
        position_extractor: PositionsExtractor[S, H],
        cutoff: float,
        start_id: int = 0,
    ) -> "NumPyHungarianObstacleIdAssignment":
        return NumPyHungarianObstacleIdAssignment(
            positions=position_extractor, cutoff=cutoff, next_id=start_id
        )

    def __call__(
        self, states: StatesT, /, *, history: HistoryT, ids: NumPyObstacleIds
    ) -> NumPyObstacleIds:
        if states.count == 0:
            return NumPyObstacleIds.empty()

        if history.horizon == 0:
            return NumPyObstacleIds.create(ids=self._allocate_ids(states.count))

        current_positions = self.positions.of_states_for_time_step(states).array
        last_positions, valid_ids = self._valid_history_from(history, ids)

        if len(valid_ids) == 0:
            return NumPyObstacleIds.create(ids=self._allocate_ids(states.count))

        current_indices, history_indices, matched = self._matching_for(
            current_positions=current_positions, last_positions=last_positions
        )

        return NumPyObstacleIds.create(
            ids=self._assign_ids(
                current_obstacle_count=states.count,
                current_indices=current_indices,
                history_indices=history_indices,
                matched=matched,
                valid_ids=valid_ids,
            )
        )

    def _valid_history_from[D_p: int = int, K: int = int](
        self, history: HistoryT, ids: NumPyObstacleIds
    ) -> tuple[Array[Dims[D_p, K]], IndexArray[Dims[K]]]:
        id_count = ids.count
        positions = self.positions.of_states(history).array

        # NOTE: The history may be padded with states for more obstacles
        # than there are IDs.
        last_positions = positions[-1, :, :id_count]

        # NOTE: Checking just the first dimension for nan is sufficient
        valid = ~np.isnan(last_positions[0])

        return last_positions[:, valid], ids.array[valid]

    def _matching_for[D_p: int = int, K_c: int = int, K_h: int = int, M: int = int](
        self,
        *,
        current_positions: Array[Dims[D_p, K_c]],
        last_positions: Array[Dims[D_p, K_h]],
    ) -> tuple[IndexArray[Dims[M]], IndexArray[Dims[M]], BoolArray[Dims[M]]]:
        distances = cdist(current_positions.T, last_positions.T)
        current_indices, history_indices = linear_sum_assignment(distances)
        matched = distances[current_indices, history_indices] <= self.cutoff

        return current_indices, history_indices, matched

    def _assign_ids[K_c: int = int, M: int = int](
        self,
        *,
        current_obstacle_count: K_c,
        current_indices: IndexArray[Dims[M]],
        history_indices: IndexArray[Dims[M]],
        matched: BoolArray[Dims[M]],
        valid_ids: IndexArray,
    ) -> IndexArray[Dims[K_c]]:
        result = np.full(current_obstacle_count, -1, dtype=np.int64)
        result[current_indices[matched]] = valid_ids[history_indices[matched]]

        unmatched = result == -1
        result[unmatched] = self._allocate_ids(np.sum(unmatched))

        return cast(IndexArray[Dims[K_c]], result)

    def _allocate_ids[K: int = int](self, count: K) -> IndexArray[Dims[K]]:
        new_ids = np.arange(self.next_id, self.next_id + count)
        self.next_id += count

        return cast(IndexArray[Dims[K]], new_ids)
