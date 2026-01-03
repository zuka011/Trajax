from dataclasses import dataclass, field

from trajax.types import Weights, ControlInputSequence, Control, Mppi


@dataclass(frozen=True)
class ControlCollector[
    StateT,
    InputSequenceT: ControlInputSequence = ControlInputSequence,
    WeightsT: Weights = Weights,
](Mppi[StateT, InputSequenceT, WeightsT]):
    inner: Mppi[StateT, InputSequenceT, WeightsT]
    _collected: list[Control[InputSequenceT, WeightsT]] = field(default_factory=list)

    @staticmethod
    def decorating[S, IS: ControlInputSequence, W: Weights](
        mppi: Mppi[S, IS, W],
    ) -> "ControlCollector[S, IS, W]":
        return ControlCollector(mppi)

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

        return control

    @property
    def collected(self) -> list[Control[InputSequenceT, WeightsT]]:
        return self._collected
