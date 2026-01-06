try:
    from beartype import beartype  # pyright: ignore[reportMissingImports]

    typechecker = beartype
except ImportError:
    typechecker = None

from typing import Literal

from numtypes import Array
from jaxtyping import jaxtyped as jaxtyping_jaxtyped, Array as JaxArray

import jax
import numpy as np
import jax.numpy as jnp

type DataType = np.dtype
type Device = Literal["cpu", "default"]

jaxtyped = jaxtyping_jaxtyped(typechecker=typechecker)
"""Wrapper around `jaxtyping.jaxtyped` that conditionally applies type checking."""


def place(array: Array | JaxArray, *, device: Device) -> JaxArray:
    """Place an array on the specified device.

    Args:
        array: The array to place.
        device: The device to place the array on. Can be "cpu" or "default".

    Returns:
        The array placed on the specified device.
    """
    match device:
        case "cpu":
            return jnp.asarray(array, device=jax.devices("cpu")[0])
        case "default":
            return jnp.asarray(array)
        case _:
            raise ValueError(f"Unsupported device: {device}")
