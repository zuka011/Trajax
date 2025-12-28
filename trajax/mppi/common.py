from trajax.types import UpdateFunction, FilterFunction


class NoUpdate(UpdateFunction):
    """Returns the nominal input unchanged."""

    def __call__[InputSequenceT](
        self, *, nominal_input: InputSequenceT, optimal_input: InputSequenceT
    ) -> InputSequenceT:
        return nominal_input


class UseOptimalControlUpdate(UpdateFunction):
    """Sets the nominal input to the optimal input."""

    def __call__[InputSequenceT](
        self, *, nominal_input: InputSequenceT, optimal_input: InputSequenceT
    ) -> InputSequenceT:
        return optimal_input


class NoFilter(FilterFunction):
    """Returns the optimal input unchanged."""

    def __call__[InputSequenceT](
        self, *, optimal_input: InputSequenceT
    ) -> InputSequenceT:
        return optimal_input
