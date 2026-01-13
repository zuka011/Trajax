from typing import Sequence, Callable, Any
from dataclasses import dataclass, field
from functools import cached_property

from trajax.types import (
    State,
    StateSequence,
    ControlInputSequence,
    Weights,
    DynamicalModel,
    Control,
    Mppi,
    StateTrajectories,
)
from trajax.collectors.access import access
from trajax.collectors.registry import (
    DataTransformer,
    IdentityTransformer,
    ModificationNotifierMixin,
    ListCollectorMixin,
    OnModifyCallback,
)


@dataclass(frozen=True)
class StateCollector[StateT = State, InputSequenceT = Any, WeightsT = Any](
    Mppi[StateT, InputSequenceT, WeightsT],
    ModificationNotifierMixin,
    ListCollectorMixin,
):
    inner: Mppi[StateT, InputSequenceT, WeightsT]
    transformer: DataTransformer[StateT, Any]
    _callbacks: list[OnModifyCallback] = field(default_factory=list)
    _collected: list[StateT] = field(default_factory=list)

    @staticmethod
    def decorating[S: State, IS, W](
        mppi: Mppi[S, IS, W],
        *,
        transformer: DataTransformer[S, Any] = IdentityTransformer(),
    ) -> "StateCollector[S, IS, W]":
        return StateCollector(mppi, transformer=transformer)

    def step(
        self,
        *,
        temperature: float,
        nominal_input: InputSequenceT,
        initial_state: StateT,
    ) -> Control[InputSequenceT, WeightsT]:
        self._collected.append(initial_state)
        self.notify()

        return self.inner.step(
            temperature=temperature,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )

    @property
    def key(self) -> str:
        return access.states.key


@dataclass(frozen=True)
class ControlCollector[
    StateT = Any,
    InputSequenceT = ControlInputSequence,
    WeightsT = Weights,
](
    Mppi[StateT, InputSequenceT, WeightsT],
    ModificationNotifierMixin,
    ListCollectorMixin,
):
    inner: Mppi[StateT, InputSequenceT, WeightsT]
    transformer: DataTransformer[Control[InputSequenceT, WeightsT]]
    _callbacks: list[OnModifyCallback] = field(default_factory=list)
    _collected: list[Control[InputSequenceT, WeightsT]] = field(default_factory=list)

    @staticmethod
    def decorating[S, IS, W](
        mppi: Mppi[S, IS, W],
        *,
        transformer: DataTransformer[Control[IS, W]] = IdentityTransformer(),
    ) -> "ControlCollector[S, IS, W]":
        return ControlCollector(mppi, transformer=transformer)

    def step(
        self,
        *,
        temperature: float,
        nominal_input: InputSequenceT,
        initial_state: StateT,
    ) -> Control[InputSequenceT, WeightsT]:
        self._collected.append(
            control := self.inner.step(
                temperature=temperature,
                nominal_input=nominal_input,
                initial_state=initial_state,
            )
        )

        # TODO: Missing notify call.

        return control

    @property
    def key(self) -> str:
        return access.controls.key


@dataclass(frozen=True)
class TrajectoryCollector[
    StateT = Any,
    TrajectoriesT = StateSequence,
    InputSequenceT = Any,
    WeightsT = Any,
](Mppi[StateT, InputSequenceT, WeightsT], ModificationNotifierMixin):
    @dataclass(frozen=True)
    class LazyTrajectories[T]:
        factory: Callable[[], StateTrajectories[T]]

        @cached_property
        def value(self) -> StateTrajectories[T]:
            return self.factory()

    inner: Mppi[StateT, InputSequenceT, WeightsT]
    model: DynamicalModel[StateT, TrajectoriesT, Any, InputSequenceT, Any]
    transformer: DataTransformer[StateTrajectories[TrajectoriesT]]
    _callbacks: list[OnModifyCallback] = field(default_factory=list)
    _collected: list[LazyTrajectories[TrajectoriesT]] = field(default_factory=list)

    @staticmethod
    def decorating[S, SS, IS, W](
        mppi: Mppi[S, IS, W],
        *,
        model: DynamicalModel[S, SS, Any, IS, Any],
        transformer: DataTransformer[StateTrajectories[SS]] = IdentityTransformer(),
    ) -> "TrajectoryCollector[S, SS, IS, W]":
        return TrajectoryCollector(mppi, model=model, transformer=transformer)

    def step(
        self,
        *,
        temperature: float,
        nominal_input: InputSequenceT,
        initial_state: StateT,
    ) -> Control[InputSequenceT, WeightsT]:
        control = self.inner.step(
            temperature=temperature,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )

        self._collected.append(self._lazy_trajectories(control, initial_state))

        # TODO: Missing notify call.

        return control

    def collected(self, *, take: int) -> Sequence[StateTrajectories[TrajectoriesT]]:
        return [it.value for it in self._collected[:take]]

    @property
    def key(self) -> str:
        return access.trajectories.key

    @property
    def count(self) -> int:
        return len(self._collected)

    def _lazy_trajectories(
        self, control: Control[InputSequenceT, WeightsT], initial_state: StateT
    ) -> LazyTrajectories[TrajectoriesT]:
        def trajectories() -> StateTrajectories[TrajectoriesT]:
            return StateTrajectories(
                optimal=self.model.forward(control.optimal, state=initial_state),
                nominal=self.model.forward(control.nominal, state=initial_state),
            )

        return self.LazyTrajectories(trajectories)
