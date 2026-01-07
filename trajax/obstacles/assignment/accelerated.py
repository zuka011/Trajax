from dataclasses import dataclass

from trajax.types import (
    DataType,
    ObstacleIdAssignment,
    ObstacleStatesForTimeStep,
    ObstacleStatesHistory,
    JaxObstaclePositionsForTimeStep,
    JaxObstaclePositions,
    JaxObstaclePositionExtractor,
    NumPyObstaclePositionsForTimeStep,
    NumPyObstaclePositions,
    NumPyObstaclePositionExtractor,
)
from trajax.obstacles.history.accelerated import JaxObstacleIds
from trajax.obstacles.history.basic import NumPyObstacleIds
from trajax.obstacles.assignment.basic import NumPyHungarianObstacleIdAssignment

from numtypes import Array, Dims

import numpy as np
import jax.numpy as jnp


type JaxPositionsExtractor[StatesT, HistoryT] = JaxObstaclePositionExtractor[
    StatesT, HistoryT, JaxObstaclePositionsForTimeStep, JaxObstaclePositions
]


@dataclass(frozen=True)
class NumPyAdaptedObstaclePositionsForTimeStep[D_p: int, K: int](
    NumPyObstaclePositionsForTimeStep[D_p, K]
):
    _array: Array[Dims[D_p, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_p, K]]:
        return self.array

    @property
    def dimension(self) -> D_p:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_p, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyAdaptedObstaclePositions[T: int, D_p: int, K: int](
    NumPyObstaclePositions[T, D_p, K]
):
    _array: Array[Dims[T, D_p, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_p, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_p:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_p, K]]:
        return self._array


@dataclass(kw_only=True)
class PositionExtractorAdapter[StatesT, HistoryT](
    NumPyObstaclePositionExtractor[
        StatesT, HistoryT, NumPyObstaclePositionsForTimeStep, NumPyObstaclePositions
    ]
):
    inner: JaxObstaclePositionExtractor[
        StatesT, HistoryT, JaxObstaclePositionsForTimeStep, JaxObstaclePositions
    ]

    @staticmethod
    def adapt(
        extractor: JaxPositionsExtractor[StatesT, HistoryT],
    ) -> "PositionExtractorAdapter[StatesT, HistoryT]":
        return PositionExtractorAdapter(inner=extractor)

    def of_states_for_time_step(
        self, states: StatesT, /
    ) -> NumPyObstaclePositionsForTimeStep:
        return NumPyAdaptedObstaclePositionsForTimeStep(
            np.asarray(self.inner.of_states_for_time_step(states))
        )

    def of_states(self, states: HistoryT, /) -> NumPyObstaclePositions:
        return NumPyAdaptedObstaclePositions(np.asarray(self.inner.of_states(states)))


@dataclass(frozen=True)
class JaxHungarianObstacleIdAssignment[
    StatesT: ObstacleStatesForTimeStep,
    HistoryT: ObstacleStatesHistory,
](ObstacleIdAssignment[StatesT, JaxObstacleIds, HistoryT]):
    # NOTE: Internally the Hungarian assignment is still done using SciPy and NumPy.
    inner: NumPyHungarianObstacleIdAssignment[StatesT, HistoryT]

    @staticmethod
    def create[S: ObstacleStatesForTimeStep, H: ObstacleStatesHistory](
        *,
        position_extractor: JaxPositionsExtractor[S, H],
        cutoff: float,
        start_id: int = 0,
    ) -> "JaxHungarianObstacleIdAssignment":
        return JaxHungarianObstacleIdAssignment(
            inner=NumPyHungarianObstacleIdAssignment.create(
                position_extractor=PositionExtractorAdapter.adapt(position_extractor),
                cutoff=cutoff,
                start_id=start_id,
            )
        )

    def __call__(
        self, states: StatesT, /, *, history: HistoryT, ids: JaxObstacleIds
    ) -> JaxObstacleIds:
        return JaxObstacleIds.create(
            ids=jnp.asarray(
                self.inner(
                    states, history=history, ids=NumPyObstacleIds(np.asarray(ids))
                )
            )
        )
