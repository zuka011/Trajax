from typing import Protocol, Sequence


class CostSumFunction[CostsT](Protocol):
    def __call__(self, costs: Sequence[CostsT], *, initial: CostsT) -> CostsT:
        """Sums an iterable of costs into a single cost."""
        ...
