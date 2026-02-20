from faran.types import jaxtyped

from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp


@jax.jit
@jaxtyped
def replace_missing(
    *,
    x: Float[JaxArray, "*S"],
    y: Float[JaxArray, "*S"],
    heading: Float[JaxArray, "*S"],
) -> tuple[Float[JaxArray, "*S"], Float[JaxArray, "*S"], Float[JaxArray, "*S"]]:
    return (
        jnp.nan_to_num(x, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf),
        jnp.nan_to_num(y, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf),
        jnp.nan_to_num(heading, nan=0.0, posinf=0.0, neginf=0.0),
    )
