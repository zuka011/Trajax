from typing import cast, Self
from dataclasses import dataclass
from functools import cached_property

from trajax.types import DataType, jaxtyped
from trajax.obstacles.accelerated import JaxObstacleStatesForTimeStep, JaxObstacleStates
from trajax.obstacles.history.basic import (
    NumPyObstacleIds,
    NumPyObstacleStatesRunningHistory,
)

from numtypes import IndexArray, Dims
from jaxtyping import Array as JaxArray, Int, Num

import numpy as np
import jax.numpy as jnp


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacleIds[K: int]:
    _ids: Int[JaxArray, "K"]

    @staticmethod
    def create[K_: int](
        *,
        ids: Num[JaxArray, "K"],
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacleIds[K_]":
        return JaxObstacleIds(_ids=ids.astype(jnp.int32))

    def __array__(self, dtype: DataType | None = None) -> IndexArray[Dims[K]]:
        return self._numpy_array

    def numpy(self) -> NumPyObstacleIds[K]:
        return NumPyObstacleIds.create(ids=self._numpy_array)

    @property
    def count(self) -> K:
        return cast(K, self._ids.shape[0])

    @property
    def array(self) -> Int[JaxArray, "K"]:
        return self._ids

    @property
    def _numpy_array(self) -> IndexArray[Dims[K]]:
        return np.asarray(self._ids, dtype=np.intp)


@dataclass(kw_only=True, frozen=True)
class JaxObstacleStatesRunningHistory[H: int, K: int]:
    # NOTE: Internally uses NumPy implementation, since it's faster in most cases.
    history: NumPyObstacleStatesRunningHistory[H, K]

    @staticmethod
    def empty[H_: int = int, K_: int = int](
        *, horizon: H_ | None = None, obstacle_count: K_ | None = None
    ) -> "JaxObstacleStatesRunningHistory[H_, K_]":
        return JaxObstacleStatesRunningHistory(
            history=NumPyObstacleStatesRunningHistory.empty(
                horizon=horizon, obstacle_count=obstacle_count
            )
        )

    @staticmethod
    def single[H_: int = int, K_: int = int](
        observation: JaxObstacleStatesForTimeStep,
        *,
        horizon: H_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacleStatesRunningHistory[H_, K_]":
        return JaxObstacleStatesRunningHistory(
            history=NumPyObstacleStatesRunningHistory.single(
                observation=observation.numpy(),
                horizon=horizon,
                obstacle_count=obstacle_count,
            )
        )

    def last(self) -> JaxObstacleStatesForTimeStep[K]:
        return self._get.last()

    def get(self) -> JaxObstacleStates[int, K]:
        return self._get

    def ids(self) -> JaxObstacleIds[K]:
        return self._ids

    def append[N: int](
        self,
        observation: JaxObstacleStatesForTimeStep[N],
        *,
        ids: "JaxObstacleIds[N] | None" = None,
    ) -> Self:
        return self.__class__(
            history=self.history.append(
                observation=observation.numpy(),
                ids=ids.numpy() if ids is not None else None,
            )
        )

    @property
    def horizon(self) -> int:
        return self.history.horizon

    @property
    def count(self) -> K:
        return self.history.count

    @cached_property
    def _get(self) -> JaxObstacleStates[int, K]:
        return JaxObstacleStates.wrap(jnp.asarray(self.history.get().array))

    @cached_property
    def _ids(self) -> JaxObstacleIds[K]:
        return JaxObstacleIds.create(
            ids=jnp.asarray(self.history.ids().array), obstacle_count=self.count
        )
