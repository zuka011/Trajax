from typing import Sequence, Self, NamedTuple, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import DataType
from trajax.obstacles.basic import NumPyObstacleStatesForTimeStep, NumPyObstacleStates

from numtypes import NumberArray, IndexArray, Dims, D, shape_of

import numpy as np


@dataclass(frozen=True)
class NumPyObstacleIds[K: int]:
    _array: IndexArray[Dims[K]]

    @staticmethod
    def empty() -> "NumPyObstacleIds[D[0]]":
        return NumPyObstacleIds(np.array([], dtype=np.intp))

    @staticmethod
    def create[K_: int](*, ids: NumberArray[Dims[K_]]) -> "NumPyObstacleIds[K_]":
        return NumPyObstacleIds(ids.astype(np.intp))

    def __array__(self, dtype: DataType | None = None) -> IndexArray[Dims[K]]:
        return self.array

    @property
    def count(self) -> K:
        return self.array.shape[0]

    @property
    def array(self) -> IndexArray[Dims[K]]:
        return self._array


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStatesRunningHistory[H: int, K: int]:
    class Entry(NamedTuple):
        states: NumPyObstacleStatesForTimeStep
        ids: NumPyObstacleIds | None

    class MergedIds[K_: int](NamedTuple):
        all: NumPyObstacleIds
        recent: NumPyObstacleIds[K_]

    history: list[Entry]
    fixed_horizon: H | None
    fixed_obstacle_count: K | None

    @staticmethod
    def empty[H_: int = int, K_: int = int](
        *, horizon: H_ | None = None, obstacle_count: K_ | None = None
    ) -> "NumPyObstacleStatesRunningHistory[H_, K_]":
        return NumPyObstacleStatesRunningHistory(
            history=[], fixed_horizon=horizon, fixed_obstacle_count=obstacle_count
        )

    @staticmethod
    def single[H_: int = int, K_: int = int](
        observation: NumPyObstacleStatesForTimeStep,
        *,
        horizon: H_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "NumPyObstacleStatesRunningHistory[H_, K_]":
        return NumPyObstacleStatesRunningHistory(
            history=[NumPyObstacleStatesRunningHistory.Entry(observation, None)],
            fixed_horizon=horizon,
            fixed_obstacle_count=obstacle_count,
        )

    def last(self) -> NumPyObstacleStatesForTimeStep[K]:
        assert self.horizon > 0, "Cannot get last state from empty history."

        return self.history[-1].states

    def get(self) -> NumPyObstacleStates[int, K]:
        return (
            self._combined_history
            if self.horizon > 0
            else NumPyObstacleStates.empty(
                horizon=self.fixed_horizon or 0,
                obstacle_count=self.fixed_obstacle_count or 0,
            )
        )

    def ids(self) -> NumPyObstacleIds[K]:
        return (
            ids.recent
            if (ids := self._merged_ids) is not None
            else NumPyObstacleIds.create(ids=np.array([]))
        )

    def append[N: int](
        self,
        observation: NumPyObstacleStatesForTimeStep,
        *,
        ids: NumPyObstacleIds | None = None,
    ) -> Self:
        assert ids is None or ids.count == observation.count, (
            f"The number of IDs ({ids.count}) does not match "
            f"the number of obstacles in the observation ({observation.count})."
        )

        assert (
            self.fixed_obstacle_count is None
            or observation.count <= self.fixed_obstacle_count
        ), (
            f"Cannot append observation with {observation.count} obstacles to history. "
            f"The obstacle count is fixed to {self.fixed_obstacle_count}."
        )

        entries = self.history + [self.Entry(observation, ids)]

        return self.__class__(
            history=entries[-self.fixed_horizon :]
            if self.fixed_horizon is not None
            else entries,
            fixed_horizon=self.fixed_horizon,
            fixed_obstacle_count=self.fixed_obstacle_count,
        )

    @property
    def horizon(self) -> int:
        return len(self.history)

    @property
    def count(self) -> K:
        return self._count

    @cached_property
    def _count(self) -> K:
        if self.fixed_obstacle_count is not None:
            return self.fixed_obstacle_count

        return ids.all.count if (ids := self._merged_ids) is not None else cast(K, 0)

    @cached_property
    def _full_horizon(self) -> int:
        if self.fixed_horizon is not None:
            return self.fixed_horizon

        return len(self.history)

    @cached_property
    def _combined_history(self) -> NumPyObstacleStates[int, K]:
        if (ids := self._merged_ids) is None:
            return NumPyObstacleStates.wrap(
                np.stack([entry.states.array for entry in self.history], axis=0)
            )

        return combine_history(
            recent_ids=ids.recent,
            history=self.history,
            horizon=self._full_horizon,
            dimension=self.history[0].states.dimension,
            obstacle_count=self.count,
        )

    @cached_property
    def _merged_ids(self) -> "NumPyObstacleStatesRunningHistory.MergedIds[K]  | None":
        if all(entry.ids is None for entry in self.history):
            return

        # NOTE: A dict is used here to maintain insertion order while ensuring uniqueness.
        seen_ids: dict[int, None] = {}

        for entry in self.history:
            assert entry.ids is not None, (
                f"Missing IDs in history entry {entry} while others have IDs."
            )

            seen_ids.update({int(id_): None for id_ in entry.ids.array})

        all_ids = list(seen_ids.keys())
        recent_ids = (
            all_ids[-self.fixed_obstacle_count :]
            if self.fixed_obstacle_count is not None
            else all_ids
        )

        return self.MergedIds(
            all=NumPyObstacleIds.create(ids=np.sort(np.array(all_ids, dtype=np.intp))),
            recent=NumPyObstacleIds.create(
                ids=np.sort(np.array(recent_ids, dtype=np.intp))
            ),
        )


def combine_history[H: int, K: int](
    *,
    recent_ids: NumPyObstacleIds[K],
    history: Sequence[NumPyObstacleStatesRunningHistory.Entry],
    horizon: H,
    dimension: int,
    obstacle_count: K,
) -> NumPyObstacleStates[H, K]:
    recent_id_count = recent_ids.count
    time_padding = horizon - len(history)
    output = np.full((output_shape := (horizon, dimension, obstacle_count)), np.nan)

    for t, entry in enumerate(history):
        assert entry.ids is not None, (
            f"Missing IDs in history entry {entry} while others have IDs."
        )

        positions = np.clip(
            np.searchsorted(recent_ids.array, entry.ids.array), 0, recent_id_count - 1
        )

        valid_mask = (positions < recent_id_count) & (
            recent_ids.array[positions] == entry.ids.array
        )

        valid_positions = positions[valid_mask]
        output[time_padding + t, :, valid_positions] = entry.states.array[
            :, valid_mask
        ].T

    assert shape_of(output, matches=output_shape)

    return NumPyObstacleStates.wrap(output)  # type: ignore # TODO: Refactor this!
