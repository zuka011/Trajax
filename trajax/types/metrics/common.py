from typing import Protocol

from trajax.types.collectors import SimulationData


class Metric[ResultT](Protocol):
    def compute(self, data: SimulationData) -> ResultT:
        """Computes the metric based on the provided simulation data."""
        ...

    @property
    def name(self) -> str:
        """The name of the metric."""
        ...
