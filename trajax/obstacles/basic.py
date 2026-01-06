from typing import Sequence, Self, NamedTuple, Final, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    DataType,
    D_o,
    D_O,
    SampledObstacleStates,
    ObstacleStates,
)

from numtypes import Array, NumberArray, IndexArray, Dims, D, shape_of

import numpy as np


type ObstacleCovarianceArray[T: int = int, K: int = int] = Array[Dims[T, D_o, D_o, K]]


@dataclass(kw_only=True, frozen=True)
class NumPySampledObstacleStates[T: int, K: int, N: int](
    SampledObstacleStates[T, K, N]
):
    _x: Array[Dims[T, K, N]]
    _y: Array[Dims[T, K, N]]
    _heading: Array[Dims[T, K, N]]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *,
        x: Array[Dims[T_, K_, N_]],
        y: Array[Dims[T_, K_, N_]],
        heading: Array[Dims[T_, K_, N_]],
    ) -> "NumPySampledObstacleStates[T_, K_, N_]":
        return NumPySampledObstacleStates(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return np.stack([self._x, self._y, self._heading], axis=1)

    def x(self) -> Array[Dims[T, K, N]]:
        return self._x

    def y(self) -> Array[Dims[T, K, N]]:
        return self._y

    def heading(self) -> Array[Dims[T, K, N]]:
        return self._heading


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStates[T: int, K: int](
    ObstacleStates[T, K, NumPySampledObstacleStates[T, K, D[1]]]
):
    _x: Array[Dims[T, K]]
    _y: Array[Dims[T, K]]
    _heading: Array[Dims[T, K]]
    _covariance: ObstacleCovarianceArray[T, K] | None

    @staticmethod
    def empty[T_: int, K_: int = D[0]](
        *, horizon: T_, obstacle_count: K_ = 0
    ) -> "NumPyObstacleStates[T_, K_]":
        """Creates obstacle states for zero obstacles over the given time horizon."""
        empty = np.empty((horizon, obstacle_count))

        assert shape_of(empty, matches=(horizon, obstacle_count))

        return NumPyObstacleStates.create(x=empty, y=empty, heading=empty)

    @staticmethod
    def sampled[T_: int, K_: int, N_: int](  # type: ignore
        *,
        x: Array[Dims[T_, K_, N_]],
        y: Array[Dims[T_, K_, N_]],
        heading: Array[Dims[T_, K_, N_]],
    ) -> NumPySampledObstacleStates[T_, K_, N_]:
        return NumPySampledObstacleStates.create(x=x, y=y, heading=heading)

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Array[Dims[T_, K_]],
        y: Array[Dims[T_, K_]],
        heading: Array[Dims[T_, K_]],
        covariance: ObstacleCovarianceArray[T_, K_] | None = None,
    ) -> "NumPyObstacleStates[T_, K_]":
        return NumPyObstacleStates(_x=x, _y=y, _heading=heading, _covariance=covariance)

    @staticmethod
    def of_states[T_: int, K_: int](
        obstacle_states: Sequence["NumPyObstacleStates[int, K_]"],
        *,
        horizon: T_ | None = None,
    ) -> "NumPyObstacleStates[T_, K_]":
        assert horizon is None or len(obstacle_states) == horizon, (
            f"Expected horizon {horizon}, but got {len(obstacle_states)} obstacle states."
        )

        x = np.stack([states.x()[0] for states in obstacle_states], axis=0)
        y = np.stack([states.y()[0] for states in obstacle_states], axis=0)
        heading = np.stack([states.heading()[0] for states in obstacle_states], axis=0)

        return NumPyObstacleStates.create(x=x, y=y, heading=heading)

    @staticmethod
    def for_time_step[K_: int](
        *, x: Array[Dims[K_]], y: Array[Dims[K_]], heading: Array[Dims[K_]]
    ) -> "NumPyObstacleStatesForTimeStep[K_]":
        return NumPyObstacleStatesForTimeStep.create(x=x, y=y, heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=1)

    def x(self) -> Array[Dims[T, K]]:
        return self._x

    def y(self) -> Array[Dims[T, K]]:
        return self._y

    def heading(self) -> Array[Dims[T, K]]:
        return self._heading

    def covariance(self) -> ObstacleCovarianceArray[T, K] | None:
        return self._covariance

    def single(self) -> NumPySampledObstacleStates[T, K, D[1]]:
        return NumPySampledObstacleStates.create(
            x=self._x[..., np.newaxis],
            y=self._y[..., np.newaxis],
            heading=self._heading[..., np.newaxis],
        )

    def at(self, *, time_step: int) -> "NumPyObstacleStatesForTimeStep[K]":
        return NumPyObstacleStatesForTimeStep.create(
            x=self._x[time_step],
            y=self._y[time_step],
            heading=self._heading[time_step],
        )

    @property
    def horizon(self) -> T:
        return self._x.shape[0]

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return self._x.shape[1]

    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=1)


@dataclass(frozen=True)
class NumPyObstacleIds[K: int]:
    _array: IndexArray[Dims[K]]

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
class NumPyObstacleStatesForTimeStep[K: int]:
    _x: Array[Dims[K]]
    _y: Array[Dims[K]]
    _heading: Array[Dims[K]]

    @staticmethod
    def create[K_: int](
        *,
        x: Array[Dims[K_]],
        y: Array[Dims[K_]],
        heading: Array[Dims[K_]],
    ) -> "NumPyObstacleStatesForTimeStep[K_]":
        return NumPyObstacleStatesForTimeStep(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=0)

    @property
    def x(self) -> Array[Dims[K]]:
        return self._x

    @property
    def y(self) -> Array[Dims[K]]:
        return self._y

    @property
    def heading(self) -> Array[Dims[K]]:
        return self._heading

    @property
    def count(self) -> K:
        return self._x.shape[0]


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStatesRunningHistory[K: int]:
    LARGE_INTEGER: Final = 2**20

    class Entry(NamedTuple):
        states: NumPyObstacleStatesForTimeStep
        ids: NumPyObstacleIds | None

    class MergedIds(NamedTuple):
        all: NumPyObstacleIds
        recent: NumPyObstacleIds

    history: list[Entry]
    _obstacle_count: K | None = None

    @staticmethod
    def empty[K_: int = int](
        *, obstacle_count: K_ | None = None
    ) -> "NumPyObstacleStatesRunningHistory[K_]":
        return NumPyObstacleStatesRunningHistory(
            history=[], _obstacle_count=obstacle_count
        )

    @staticmethod
    def single[K_: int](
        step: NumPyObstacleStatesForTimeStep[K_],
    ) -> "NumPyObstacleStatesRunningHistory[K_]":
        return NumPyObstacleStatesRunningHistory(
            history=[NumPyObstacleStatesRunningHistory.Entry(step, None)],
            _obstacle_count=step.count,
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[int, D_o, K]]:
        return self.array

    def last(self) -> NumPyObstacleStatesForTimeStep[K]:
        assert self.horizon > 0, "Cannot get last state from empty history."

        return self.history[-1].states

    def append[N: int](
        self,
        step: NumPyObstacleStatesForTimeStep[N],
        *,
        ids: NumPyObstacleIds[N] | None = None,
    ) -> Self:
        assert step.count <= self.capacity, (
            f"Cannot append step with {step.count} obstacles to history "
            f"with capacity {self.capacity}."
        )

        return self.__class__(
            history=self.history + [self.Entry(step, ids)],
            _obstacle_count=self._obstacle_count,
        )

    def get(self) -> NumPyObstacleStates[int, K]:
        return (
            self._combined_history
            if self.horizon > 0
            else NumPyObstacleStates.empty(horizon=0, obstacle_count=self.count)
        )

    def x(self) -> Array[Dims[int, K]]:
        return self._combined_history.x()

    def y(self) -> Array[Dims[int, K]]:
        return self._combined_history.y()

    def heading(self) -> Array[Dims[int, K]]:
        return self._combined_history.heading()

    @property
    def horizon(self) -> int:
        return len(self.history)

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return self._count

    @property
    def array(self) -> Array[Dims[int, D_o, K]]:
        return self._array

    @property
    def capacity(self) -> int:
        return (
            self._obstacle_count
            if self._obstacle_count is not None
            else self.LARGE_INTEGER
        )

    @cached_property
    def _count(self) -> K:
        if self._obstacle_count is not None:
            return self._obstacle_count

        return ids.all.count if (ids := self._merged_ids) is not None else cast(K, 0)

    @cached_property
    def _array(self) -> Array[Dims[int, D_o, K]]:
        return np.stack([self.x(), self.y(), self.heading()], axis=1)

    @cached_property
    def _combined_history(self) -> NumPyObstacleStates[int, K]:
        if (ids := self._merged_ids) is None:
            return NumPyObstacleStates.create(
                x=np.stack([entry.states.x for entry in self.history], axis=0),
                y=np.stack([entry.states.y for entry in self.history], axis=0),
                heading=np.stack(
                    [entry.states.heading for entry in self.history], axis=0
                ),
            )

        return combine_history(
            recent_ids=ids.recent, history=self.history, obstacle_count=self.count
        )

    @cached_property
    def _merged_ids(self) -> "NumPyObstacleStatesRunningHistory.MergedIds  | None":
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
        recent_ids = all_ids[-self.capacity :]

        return self.MergedIds(
            all=NumPyObstacleIds.create(ids=np.sort(np.array(all_ids, dtype=np.intp))),
            recent=NumPyObstacleIds.create(
                ids=np.sort(np.array(recent_ids, dtype=np.intp))
            ),
        )


def combine_history[K: int](
    *,
    recent_ids: NumPyObstacleIds,
    history: Sequence[NumPyObstacleStatesRunningHistory.Entry],
    obstacle_count: K,
) -> NumPyObstacleStates[int, K]:
    recent_id_count = recent_ids.count
    x = np.full(out_shape := (len(history), obstacle_count), np.inf)
    y = np.full(out_shape, np.inf)
    heading = np.full(out_shape, np.inf)

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
        x[t, valid_positions] = entry.states.x[valid_mask]
        y[t, valid_positions] = entry.states.y[valid_mask]
        heading[t, valid_positions] = entry.states.heading[valid_mask]

    assert shape_of(x, matches=(-1, obstacle_count))
    assert shape_of(y, matches=(-1, obstacle_count))
    assert shape_of(heading, matches=(-1, obstacle_count))

    return NumPyObstacleStates.create(x=x, y=y, heading=heading)
