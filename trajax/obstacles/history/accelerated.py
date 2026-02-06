from typing import Self, Protocol, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    DataType,
    jaxtyped,
    NumPyObstacleStatesForTimeStep,
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
)
from trajax.obstacles.history.basic import (
    NumPyObstacleIds,
    NumPyObstacleStatesRunningHistory,
)

from numtypes import IndexArray, Array, Dims
from jaxtyping import Array as JaxArray, Float, Int, Num

import numpy as np
import jax.numpy as jnp


class JaxObstacleStateCreator[StatesT](Protocol):
    """Protocol for creating obstacle state objects from JAX arrays."""

    def wrap(self, states: Float[JaxArray, "T D_o K"], /) -> StatesT:
        """Wraps a JAX array into the appropriate obstacle states type."""
        ...

    def empty(self, *, horizon: int, obstacle_count: int) -> StatesT:
        """Creates empty obstacle states with the given horizon and obstacle count."""
        ...


@dataclass(frozen=True)
class JaxObstacleStateCreatorAdapter[StatesT]:
    """Adapts a JAX obstacle state creator to accept NumPy arrays."""

    _creator: JaxObstacleStateCreator[StatesT]

    def wrap[T: int = int, D_o: int = int, K: int = int](
        self, states: Array[Dims[T, D_o, K]]
    ) -> StatesT:
        return self._creator.wrap(jnp.asarray(states))

    def empty(self, *, horizon: int, obstacle_count: int) -> StatesT:
        return self._creator.empty(horizon=horizon, obstacle_count=obstacle_count)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacleIds[K: int]:
    """JAX container for integer obstacle identifiers."""

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
class JaxObstacleStatesRunningHistory[
    StatesT: JaxObstacleStates,
    StatesForTimeStepT: JaxObstacleStatesForTimeStep,
]:
    """JAX sliding window history of obstacle states, delegating to NumPy internally."""

    # NOTE: Internally uses NumPy implementation, since it's faster in most cases.
    history: NumPyObstacleStatesRunningHistory[StatesT, NumPyObstacleStatesForTimeStep]

    @staticmethod
    def empty[
        S: JaxObstacleStates,
        STS: JaxObstacleStatesForTimeStep = JaxObstacleStatesForTimeStep,
    ](
        *,
        creator: JaxObstacleStateCreator[S],
        horizon: int | None = None,
        obstacle_count: int | None = None,
    ) -> "JaxObstacleStatesRunningHistory[S, STS]":
        return JaxObstacleStatesRunningHistory(
            history=NumPyObstacleStatesRunningHistory.empty(
                creator=JaxObstacleStateCreatorAdapter(creator),
                horizon=horizon,
                obstacle_count=obstacle_count,
            )
        )

    @staticmethod
    def single[S: JaxObstacleStates, STS: JaxObstacleStatesForTimeStep](
        observation: STS,
        *,
        creator: JaxObstacleStateCreator[S],
        horizon: int | None = None,
        obstacle_count: int | None = None,
    ) -> "JaxObstacleStatesRunningHistory[S, STS]":
        return JaxObstacleStatesRunningHistory(
            history=NumPyObstacleStatesRunningHistory.single(
                observation=observation.numpy(),
                creator=JaxObstacleStateCreatorAdapter(creator),
                horizon=horizon,
                obstacle_count=obstacle_count,
            )
        )

    def last(self) -> StatesForTimeStepT:
        return self._get.last()

    def get(self) -> StatesT:
        return self._get

    def ids(self) -> JaxObstacleIds:
        return self._ids

    def append(
        self, observation: StatesForTimeStepT, *, ids: JaxObstacleIds | None = None
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
    def count(self) -> int:
        return self.history.count

    @cached_property
    def _get(self) -> StatesT:
        return self.history.get()

    @cached_property
    def _ids(self) -> JaxObstacleIds:
        return JaxObstacleIds.create(
            ids=jnp.asarray(self.history.ids().array), obstacle_count=self.count
        )
