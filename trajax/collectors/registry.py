from warnings import warn
from typing import Protocol, Sequence, Mapping, Any, runtime_checkable
from dataclasses import dataclass

from trajax.types import SimulationData


class NoCollectedDataWarning(RuntimeWarning):
    """Warning raised when no data has been collected by a registered collector."""


@runtime_checkable
class SimpleCollector[T](Protocol):
    @property
    def collected(self) -> Sequence[T]:
        """Returns the collected data sequence for each time step."""
        ...

    @property
    def key(self) -> str:
        """Returns the key of the data being collected."""
        ...


@runtime_checkable
class DataTransformer[InputT, OutputT](Protocol):
    def __call__(self, data: Sequence[InputT]) -> OutputT:
        """Transforms the collected data sequence into the desired output format."""
        ...


type Collector = SimpleCollector | tuple[SimpleCollector, DataTransformer]


@dataclass(frozen=True)
class CollectorRegistry:
    collectors_by_name: Mapping[str, Collector | None]

    @staticmethod
    def of(**kwargs: Collector | None) -> "CollectorRegistry":
        return CollectorRegistry(collectors_by_name=kwargs)

    @property
    def data(self) -> SimulationData:
        return SimulationData.create(
            **{
                self._key_from(collector, name=name): data
                for name, collector in self.collectors_by_name.items()
                if collector is not None
                and (data := self._data_from(collector, name=name)) is not None
            }
        )

    def _key_from(self, collector: Collector, *, name: str) -> str:
        match collector:
            case (SimpleCollector() as simple, _):
                return simple.key
            case SimpleCollector() as simple:
                return simple.key
            case _:
                raise TypeError(
                    f"Unexpected collector '{name}' of type: {type(collector)}"
                )

    def _data_from(self, collector: Collector, *, name: str) -> Any:
        match collector:
            case (SimpleCollector() as simple, DataTransformer() as transformer):
                if len(collected := simple.collected) == 0:
                    warn(
                        f"The collector '{name}' did not collect any data.",
                        NoCollectedDataWarning,
                    )
                    return

                return transformer(collected)
            case SimpleCollector() as simple:
                return simple.collected
            case _:
                raise TypeError(
                    f"Unexpected collector '{name}' of type: {type(collector)}"
                )
