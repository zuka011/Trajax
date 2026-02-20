from typing import Protocol, Sequence, Self, NamedTuple
from dataclasses import dataclass
from functools import cached_property

from faran.types import DataType, NumPyObstacleStatesForTimeStep

from numtypes import NumberArray, IndexArray, Array, Dims, D, shape_of

import numpy as np


class NumPyObstacleStateCreator[StatesT](Protocol):
    """Protocol for creating obstacle state objects from NumPy arrays."""

    def wrap[T: int = int, D_o: int = int, K: int = int](
        self, states: Array[Dims[T, D_o, K]], /
    ) -> StatesT:
        """Wraps a NumPy array into the appropriate obstacle states type."""
        ...

    def empty(self, *, horizon: int, obstacle_count: int) -> StatesT:
        """Creates empty obstacle states with the given horizon and obstacle count."""
        ...


class HistoryEntry[StatesForTimeStepT](Protocol):
    """Single entry in an obstacle state history, pairing states with optional IDs."""

    @property
    def states(self) -> StatesForTimeStepT:
        """The obstacle states at this time step."""
        ...

    @property
    def ids(self) -> "NumPyObstacleIds | None":
        """The obstacle IDs at this time step, if available."""
        ...


@dataclass(frozen=True)
class NumPyObstacleIds[K: int]:
    """NumPy container for integer obstacle identifiers."""

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
class NumPyObstacleStatesRunningHistory[
    StatesT,
    StatesForTimeStepT: NumPyObstacleStatesForTimeStep,
]:
    """NumPy sliding window history of obstacle states with ID-based alignment."""

    class Entry[STS: NumPyObstacleStatesForTimeStep](NamedTuple):
        states: STS
        ids: NumPyObstacleIds | None

    class MergedIds(NamedTuple):
        all: NumPyObstacleIds
        recent: NumPyObstacleIds

    history: list[Entry[StatesForTimeStepT]]
    creator: NumPyObstacleStateCreator[StatesT]
    fixed_horizon: int | None
    fixed_obstacle_count: int | None

    @staticmethod
    def empty[S, STS: NumPyObstacleStatesForTimeStep = NumPyObstacleStatesForTimeStep](
        *,
        creator: NumPyObstacleStateCreator[S],
        horizon: int | None = None,
        obstacle_count: int | None = None,
    ) -> "NumPyObstacleStatesRunningHistory[S, STS]":
        return NumPyObstacleStatesRunningHistory(
            history=[],
            creator=creator,
            fixed_horizon=horizon,
            fixed_obstacle_count=obstacle_count,
        )

    @staticmethod
    def single[S, STS: NumPyObstacleStatesForTimeStep](
        observation: STS,
        *,
        creator: NumPyObstacleStateCreator[S],
        horizon: int | None = None,
        obstacle_count: int | None = None,
    ) -> "NumPyObstacleStatesRunningHistory[S, STS]":
        return NumPyObstacleStatesRunningHistory(
            history=[NumPyObstacleStatesRunningHistory.Entry(observation, None)],
            creator=creator,
            fixed_horizon=horizon,
            fixed_obstacle_count=obstacle_count,
        )

    def last(self) -> StatesForTimeStepT:
        assert self.horizon > 0, "Cannot get last state from empty history."

        return self.history[-1].states

    def get(self) -> StatesT:
        return (
            self._combined_history
            if self.horizon > 0
            else self.creator.empty(
                horizon=self.fixed_horizon or 0,
                obstacle_count=self.fixed_obstacle_count or 0,
            )
        )

    def ids(self) -> NumPyObstacleIds:
        return (
            ids.recent
            if (ids := self._merged_ids) is not None
            else NumPyObstacleIds.create(ids=np.array([]))
        )

    def append(
        self, observation: StatesForTimeStepT, *, ids: NumPyObstacleIds | None = None
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
            creator=self.creator,
            fixed_horizon=self.fixed_horizon,
            fixed_obstacle_count=self.fixed_obstacle_count,
        )

    @property
    def horizon(self) -> int:
        return len(self.history)

    @property
    def count(self) -> int:
        return self._count

    @cached_property
    def _count(self) -> int:
        if self.fixed_obstacle_count is not None:
            return self.fixed_obstacle_count

        return ids.all.count if (ids := self._merged_ids) is not None else 0

    @cached_property
    def _full_horizon(self) -> int:
        if self.fixed_horizon is not None:
            return self.fixed_horizon

        return len(self.history)

    @cached_property
    def _combined_history(self) -> StatesT:
        return self.creator.wrap(
            np.stack([entry.states.array for entry in self.history], axis=0)
            if (ids := self._merged_ids) is None
            else combine_history(
                recent_ids=ids.recent,
                history=self.history,
                horizon=self._full_horizon,
                obstacle_count=self.count,
            )
        )

    @cached_property
    def _merged_ids(self) -> "NumPyObstacleStatesRunningHistory.MergedIds | None":
        if all(entry.ids is None for entry in self.history):
            return None

        # NOTE: A dict is used here to maintain insertion order while ensuring uniqueness.
        seen_ids: dict[int, None] = {}

        # NOTE: We iterate in reverse to get most recent IDs first.
        for entry in reversed(self.history):
            assert entry.ids is not None, (
                f"Missing IDs in history entry {entry} while others have IDs."
            )

            for obstacle_id in entry.ids.array:
                # NOTE: We do not overwrite existing IDs to maintain most recent occurrence.
                if (typed_id := int(obstacle_id)) not in seen_ids:
                    seen_ids[typed_id] = None

        all_ids = list(seen_ids.keys())
        recent_ids = (
            all_ids[: self.fixed_obstacle_count]
            if self.fixed_obstacle_count is not None
            else all_ids
        )

        return self.MergedIds(
            all=NumPyObstacleIds.create(ids=np.sort(np.array(all_ids, dtype=np.intp))),
            recent=NumPyObstacleIds.create(
                ids=np.sort(np.array(recent_ids, dtype=np.intp))
            ),
        )


def combine_history[H: int, K: int, D_o: int = int](
    *,
    recent_ids: NumPyObstacleIds[K],
    history: Sequence[HistoryEntry[NumPyObstacleStatesForTimeStep]],
    horizon: H,
    obstacle_count: K,
) -> Array[Dims[H, D_o, K]]:
    recent_id_count = recent_ids.count
    time_padding = horizon - len(history)
    dimension = history[0].states.dimension
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

    return output
