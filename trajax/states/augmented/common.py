import asyncio
from typing import Callable, Protocol, cast
from dataclasses import dataclass

from trajax.type import DataType
from trajax.model import (
    State,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
)
from trajax.mppi import Sampler

from numtypes import Array, Dims

import numpy as np


@dataclass(kw_only=True, frozen=True)
class AugmentedState[PhysicalT: State, VirtualT: State, D_x: int = int](State[D_x]):
    physical: PhysicalT
    virtual: VirtualT
    _dimension: D_x

    @staticmethod
    def of[P: State, V: State, D_x_: int = int](
        *, physical: P, virtual: V, dimension: D_x_ | None = None
    ) -> "AugmentedState[P, V, D_x_]":
        return AugmentedState(
            physical=physical,
            virtual=virtual,
            _dimension=cast(
                D_x_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
        )

    def __post_init__(self) -> None:
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"State dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ]
        )

    @property
    def dimension(self) -> D_x:
        return self._dimension


@dataclass(kw_only=True, frozen=True)
class AugmentedStateBatch[
    PhysicalBatchT: StateBatch,
    VirtualBatchT: StateBatch,
    T: int = int,
    D_x: int = int,
    M: int = int,
](StateBatch[T, D_x, M]):
    physical: PhysicalBatchT
    virtual: VirtualBatchT
    _horizon: T
    _dimension: D_x
    _rollout_count: M

    @staticmethod
    def of[
        P: StateBatch,
        V: StateBatch,
        T_: int = int,
        D_x_: int = int,
        M_: int = int,
    ](
        *,
        physical: P,
        virtual: V,
        horizon: T_ | None = None,
        dimension: D_x_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "AugmentedStateBatch[P, V, T_, D_x_, M_]":
        return AugmentedStateBatch(
            physical=physical,
            virtual=virtual,
            _horizon=cast(
                T_,
                horizon if horizon is not None else physical.horizon,
            ),
            _dimension=cast(
                D_x_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
            _rollout_count=cast(
                M_,
                rollout_count if rollout_count is not None else physical.rollout_count,
            ),
        )

    @staticmethod
    def from_physical[
        R,
        P: StateBatch,
        V: StateBatch = StateBatch,
        T_: int = int,
        D_x_: int = int,
        M_: int = int,
    ](
        extract: Callable[[P], R],
    ) -> Callable[["AugmentedStateBatch[P, V, T_, D_x_, M_]"], R]:
        return lambda batch: extract(batch.physical)

    @staticmethod
    def from_virtual[
        R,
        V: StateBatch,
        P: StateBatch = StateBatch,
        T_: int = int,
        D_x_: int = int,
        M_: int = int,
    ](
        extract: Callable[[V], R],
    ) -> Callable[["AugmentedStateBatch[P, V, T_, D_x_, M_]"], R]:
        return lambda batch: extract(batch.virtual)

    def __post_init__(self) -> None:
        assert self.horizon == self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"expected {self.horizon}, got "
            f"{self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"State dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )
        assert (
            self.rollout_count
            == self.physical.rollout_count
            == self.virtual.rollout_count
        ), (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"expected {self.rollout_count}, got "
            f"{self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ],
            axis=1,
        )

    @property
    def horizon(self) -> T:
        return self._horizon

    @property
    def dimension(self) -> D_x:
        return self._dimension

    @property
    def rollout_count(self) -> M:
        return self._rollout_count


class AugmentedControlInputSequence[
    PhysicalT: ControlInputSequence,
    VirtualT: ControlInputSequence,
    T: int = int,
    D_u: int = int,
](ControlInputSequence[T, D_u], Protocol):
    @property
    def physical(self) -> PhysicalT:
        """Returns the physical control input sequence."""
        ...

    @property
    def virtual(self) -> VirtualT:
        """Returns the virtual control input sequence."""
        ...


@dataclass(kw_only=True, frozen=True)
class AugmentedControlInputBatch[
    PhysicalBatchT: StateBatch,
    VirtualBatchT: StateBatch,
    T: int = int,
    D_u: int = int,
    M: int = int,
](ControlInputBatch[T, D_u, M]):
    physical: PhysicalBatchT
    virtual: VirtualBatchT
    _horizon: T
    _dimension: D_u
    _rollout_count: M

    @staticmethod
    def of[
        P: StateBatch,
        V: StateBatch,
        T_: int = int,
        D_u_: int = int,
        M_: int = int,
    ](
        *,
        physical: P,
        virtual: V,
        horizon: T_ | None = None,
        dimension: D_u_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "AugmentedControlInputBatch[P, V, T_, D_u_, M_]":
        return AugmentedControlInputBatch(
            physical=physical,
            virtual=virtual,
            _horizon=cast(
                T_,
                horizon if horizon is not None else physical.horizon,
            ),
            _dimension=cast(
                D_u_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
            _rollout_count=cast(
                M_,
                rollout_count if rollout_count is not None else physical.rollout_count,
            ),
        )

    @staticmethod
    def from_virtual[
        R,
        V: ControlInputBatch,
        P: ControlInputBatch = ControlInputBatch,
        T_: int = int,
        D_u_: int = int,
        M_: int = int,
    ](
        extract: Callable[[V], R],
    ) -> Callable[["AugmentedControlInputBatch[P, V, T_, D_u_, M_]"], R]:
        return lambda batch: extract(batch.virtual)

    def __post_init__(self) -> None:
        assert self.horizon == self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"expected {self.horizon}, got "
            f"{self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"Control dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )
        assert (
            self.rollout_count
            == self.physical.rollout_count
            == self.virtual.rollout_count
        ), (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"expected {self.rollout_count}, got "
            f"{self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ],
            axis=1,
        )

    @property
    def horizon(self) -> T:
        return self._horizon

    @property
    def dimension(self) -> D_u:
        return self._dimension

    @property
    def rollout_count(self) -> M:
        return self._rollout_count


@dataclass(kw_only=True, frozen=True)
class AugmentedModel[
    PInStateT: State,
    POutStateT: State,
    PStateBatchT: StateBatch,
    PControlInputSequenceT: ControlInputSequence,
    PControlInputBatchT: ControlInputBatch,
    VInStateT: State,
    VOutStateT: State,
    VStateBatchT: StateBatch,
    VControlInputSequenceT: ControlInputSequence,
    VControlInputBatchT: ControlInputBatch,
    T: int = int,
    D_u: int = int,
    D_x: int = int,
    M: int = int,
](
    DynamicalModel[
        AugmentedState[PInStateT, VInStateT, D_x],
        AugmentedState[POutStateT, VOutStateT, D_x],
        AugmentedStateBatch[PStateBatchT, VStateBatchT, T, D_x, M],
        AugmentedControlInputSequence[
            PControlInputSequenceT, VControlInputSequenceT, T, D_u
        ],
        AugmentedControlInputBatch[PControlInputBatchT, VControlInputBatchT, T, D_u, M],
    ]
):
    physical: DynamicalModel[
        PInStateT, POutStateT, PStateBatchT, PControlInputSequenceT, PControlInputBatchT
    ]
    virtual: DynamicalModel[
        VInStateT, VOutStateT, VStateBatchT, VControlInputSequenceT, VControlInputBatchT
    ]

    @staticmethod
    def of[
        PIS: State,
        POS: State,
        PSB: StateBatch,
        PCIS: ControlInputSequence,
        PCIB: ControlInputBatch,
        VIS: State,
        VOS: State,
        VSB: StateBatch,
        VCIS: ControlInputSequence,
        VCIB: ControlInputBatch,
        T_: int = int,
        D_u_: int = int,
        D_x_: int = int,
        M_: int = int,
    ](
        *,
        physical: DynamicalModel[PIS, POS, PSB, PCIS, PCIB],
        virtual: DynamicalModel[VIS, VOS, VSB, VCIS, VCIB],
        horizon: T_ | None = None,
        control_dimension: D_u_ | None = None,
        state_dimension: D_x_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "AugmentedModel[PIS, POS, PSB, PCIS, PCIB, VIS, VOS, VSB, VCIS, VCIB, T_, D_u_, D_x_, M_]":
        return AugmentedModel(physical=physical, virtual=virtual)

    async def simulate(
        self,
        inputs: AugmentedControlInputBatch[
            PControlInputBatchT, VControlInputBatchT, T, D_u, M
        ],
        initial_state: AugmentedState[PInStateT, VInStateT, D_x],
    ) -> AugmentedStateBatch[PStateBatchT, VStateBatchT, T, D_x, M]:
        physical, virtual = await asyncio.gather(
            self.physical.simulate(
                inputs=inputs.physical, initial_state=initial_state.physical
            ),
            self.virtual.simulate(
                inputs=inputs.virtual, initial_state=initial_state.virtual
            ),
        )

        return AugmentedStateBatch.of(
            physical=physical,
            virtual=virtual,
            horizon=inputs.horizon,
            dimension=initial_state.dimension,
            rollout_count=inputs.rollout_count,
        )

    async def step(
        self,
        input: AugmentedControlInputSequence[
            PControlInputSequenceT, VControlInputSequenceT, T, D_u
        ],
        state: AugmentedState[PInStateT, VInStateT, D_x],
    ) -> AugmentedState[POutStateT, VOutStateT, D_x]:
        physical, virtual = await asyncio.gather(
            self.physical.step(input=input.physical, state=state.physical),
            self.virtual.step(input=input.virtual, state=state.virtual),
        )

        return AugmentedState.of(
            physical=physical, virtual=virtual, dimension=state.dimension
        )


@dataclass(kw_only=True, frozen=True)
class AugmentedSampler[
    PhysicalSequenceT: ControlInputSequence,
    PhysicalBatchT: ControlInputBatch,
    VirtualSequenceT: ControlInputSequence,
    VirtualBatchT: ControlInputBatch,
    T: int = int,
    D_u: int = int,
    M: int = int,
](
    Sampler[
        AugmentedControlInputSequence[PhysicalSequenceT, VirtualSequenceT, T, D_u],
        AugmentedControlInputBatch[PhysicalBatchT, VirtualBatchT, T, D_u, M],
    ]
):
    physical: Sampler[PhysicalSequenceT, PhysicalBatchT]
    virtual: Sampler[VirtualSequenceT, VirtualBatchT]

    _rollout_count: M

    @staticmethod
    def of[
        PS: ControlInputSequence,
        PB: ControlInputBatch,
        VS: ControlInputSequence,
        VB: ControlInputBatch,
        T_: int = int,
        D_u_: int = int,
        M_: int = int,
    ](
        *,
        physical: Sampler[PS, PB, M_],
        virtual: Sampler[VS, VB, M_],
        horizon: T_ | None = None,
        dimension: D_u_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "AugmentedSampler[PS, PB, VS, VB, T_, D_u_, M_]":
        return AugmentedSampler(
            physical=physical,
            virtual=virtual,
            _rollout_count=cast(
                M_,
                rollout_count if rollout_count is not None else physical.rollout_count,
            ),
        )

    def __post_init__(self) -> None:
        assert (
            self.rollout_count
            == self.physical.rollout_count
            == self.virtual.rollout_count
        ), (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"expected {self.rollout_count}, got "
            f"{self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def sample(
        self,
        *,
        around: AugmentedControlInputSequence[
            PhysicalSequenceT, VirtualSequenceT, T, D_u
        ],
    ) -> AugmentedControlInputBatch[PhysicalBatchT, VirtualBatchT, T, D_u, M]:
        physical_batch = self.physical.sample(around=around.physical)
        virtual_batch = self.virtual.sample(around=around.virtual)

        return AugmentedControlInputBatch.of(
            physical=physical_batch,
            virtual=virtual_batch,
            horizon=around.horizon,
            dimension=around.dimension,
            rollout_count=self.rollout_count,
        )

    @property
    def rollout_count(self) -> M:
        return self._rollout_count
