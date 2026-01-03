from dataclasses import dataclass

from trajax.types import jaxtyped, JaxControlInputSequence

from jaxtyping import Array as JaxArray, Float
from scipy.signal import savgol_coeffs

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class JaxSavGolFilter:
    coefficients: Float[JaxArray, "W"]

    @staticmethod
    def create(*, window_length: int, polynomial_order: int) -> "JaxSavGolFilter":
        assert window_length % 2 == 1, f"Window length must be odd, got {window_length}"
        assert 0 <= polynomial_order < window_length, (
            f"Polynomial order must be non-negative and less than window length, got {polynomial_order}"
        )

        return JaxSavGolFilter(
            coefficients=jnp.asarray(savgol_coeffs(window_length, polynomial_order))
        )

    def __call__[InputSequenceT: JaxControlInputSequence](
        self, *, optimal_input: InputSequenceT
    ) -> InputSequenceT:
        return optimal_input.similar(
            array=convolve_along_time_axis(
                optimal_input.array, coefficients=self.coefficients
            )
        )


@jax.jit
@jaxtyped
def convolve_along_time_axis[T: int, D_u: int](
    input_array: Float[JaxArray, "T D_u"], *, coefficients: Float[JaxArray, "W"]
) -> Float[JaxArray, "T D_u"]:
    half_window = coefficients.shape[0] // 2
    padded = jnp.pad(input_array, ((half_window, half_window), (0, 0)), mode="edge")

    return jax.vmap(
        lambda col: jnp.convolve(col, coefficients, mode="valid"), in_axes=1, out_axes=1
    )(padded)
