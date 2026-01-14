from warnings import warn
from typing import Protocol, Sequence, Any, Callable
from dataclasses import dataclass

from trajax.types import SimulationData


type OnModifyCallback = Callable[[], None]


class NoCollectedDataWarning(RuntimeWarning):
    """Warning raised when no data has been collected by a registered collector."""


class DataTransformer[InputT, OutputT = Any](Protocol):
    def __call__(self, data: Sequence[InputT], /) -> OutputT:
        """Transforms the collected data into the desired output format."""
        ...


class Collector[T](Protocol):
    def on_modified(self, callback: OnModifyCallback) -> None:
        """Registers a callback to be invoked when new data is collected."""
        ...

    def collected(self, *, take: int) -> Sequence[T]:
        """Returns the first `take` collected data points."""
        ...

    @property
    def count(self) -> int:
        """Returns the number of collected data points."""
        ...

    @property
    def key(self) -> str:
        """Returns the key of the data being collected."""
        ...

    @property
    def transformer(self) -> DataTransformer[T, Any]:
        """Returns the transformer function to be applied to the collected data."""
        ...


class HasCallbacks(Protocol):
    @property
    def _callbacks(self) -> list[OnModifyCallback]:
        """Returns the list of registered modification callbacks."""
        ...


class HasList[T](Protocol):
    @property
    def _collected(self) -> list[T]:
        """Returns the list of collected data."""
        ...


class IdentityTransformer[InputT]:
    def __call__(self, data: Sequence[InputT], /) -> Sequence[InputT]:
        """Returns the input data unchanged."""
        return data


class ModificationNotifierMixin:
    def notify(self: HasCallbacks) -> None:
        for callback in self._callbacks:
            callback()

    def on_modified(self: HasCallbacks, callback: OnModifyCallback) -> None:
        self._callbacks.append(callback)


class ListCollectorMixin:
    def collected[T](self: HasList[T], *, take: int) -> Sequence[T]:
        return self._collected[:take]

    @property
    def count(self: HasList[Any]) -> int:
        return len(self._collected)


@dataclass
class CollectorRegistry:
    collectors: tuple[Collector, ...]
    _data: SimulationData | None

    @staticmethod
    def of(*args: Collector) -> "CollectorRegistry":
        registry = CollectorRegistry(collectors=(), _data=None)
        registry.collectors = tuple(
            registry._with_callback(collector)
            for collector in args
            if collector is not None
        )

        return registry

    def on_modified(self, callback: OnModifyCallback) -> None:
        for collector in self.collectors:
            collector.on_modified(callback)

    @property
    def data(self) -> SimulationData:
        if self._data is None:
            self._data = self._collect_data()

        return self._data

    def _modified(self) -> None:
        self._data = None

    def _collect_data(self) -> SimulationData:
        for collector in self.collectors:
            if collector.count == 0:
                warn(
                    f"Collector for key '{collector.key}' has not collected any data.",
                    NoCollectedDataWarning,
                )

        collectors = [it for it in self.collectors if it.count > 0]
        time_step_count = min(collector.count for collector in collectors)

        return SimulationData.create(
            {
                collector.key: collector.transformer(
                    collector.collected(take=time_step_count)
                )
                for collector in collectors
            }
        )

    def _with_callback(self, collector: Collector) -> Collector:
        collector.on_modified(self._modified)
        return collector
