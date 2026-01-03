from dataclasses import dataclass

from trajax.types import NumPyControlInputSequence

from numtypes import Array, Dim1, shape_of
from scipy.signal import savgol_coeffs
from scipy.ndimage import convolve1d


@dataclass(frozen=True)
class NumPySavGolFilter:
    coefficients: Array[Dim1]

    @staticmethod
    def create(*, window_length: int, polynomial_order: int) -> "NumPySavGolFilter":
        assert window_length % 2 == 1, f"Window length must be odd, got {window_length}"
        assert 0 <= polynomial_order < window_length, (
            f"Polynomial order must be non-negative and less than window length, got {polynomial_order}"
        )

        coefficients = savgol_coeffs(window_length, polynomial_order)

        assert shape_of(coefficients, matches=(window_length,))

        return NumPySavGolFilter(coefficients=coefficients)

    def __call__[InputSequenceT: NumPyControlInputSequence](
        self, *, optimal_input: InputSequenceT
    ) -> InputSequenceT:
        filtered = convolve1d(
            optimal_input.array, weights=self.coefficients, axis=0, mode="nearest"
        )

        assert shape_of(filtered, matches=optimal_input.array.shape)

        return optimal_input.similar(array=filtered)
