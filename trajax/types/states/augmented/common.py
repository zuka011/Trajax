from typing import Protocol


class AugmentedState[P, V](Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented state."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented state."""
        ...


class AugmentedStateSequence[P, V](Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented state sequence."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented state sequence."""
        ...


class AugmentedStateBatch[P, V](Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented state batch."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented state batch."""
        ...


class AugmentedControlInputSequence[P, V](Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented control input sequence."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented control input sequence."""
        ...


class AugmentedControlInputBatch[P, V](Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented control input batch."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented control input batch."""
        ...


class AugmentedStateCreator[P, V, A](Protocol):
    def of(self, *, physical: P, virtual: V) -> A:
        """Creates an augmented state from physical and virtual parts."""
        ...


class AugmentedStateSequenceCreator[P, V, A](Protocol):
    def of(self, *, physical: P, virtual: V) -> A:
        """Creates an augmented state sequence from physical and virtual parts."""
        ...


class AugmentedStateBatchCreator[P, V, A](Protocol):
    def of(self, *, physical: P, virtual: V) -> A:
        """Creates an augmented state batch from physical and virtual parts."""
        ...


class AugmentedControlInputBatchCreator[P, V, A](Protocol):
    def of(self, *, physical: P, virtual: V) -> A:
        """Creates an augmented control input batch from physical and virtual parts."""
        ...


class HasPhysical[P](Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part."""
        ...


class HasVirtual[V](Protocol):
    @property
    def virtual(self) -> V:
        """Returns the virtual part."""
        ...
