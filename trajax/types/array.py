try:
    from beartype import beartype  # pyright: ignore[reportMissingImports]

    typechecker = beartype
except ImportError:
    typechecker = None

import numpy as np
from jaxtyping import jaxtyped as jaxtyping_jaxtyped

type DataType = np.dtype

jaxtyped = jaxtyping_jaxtyped(typechecker=typechecker)
"""Wrapper around `jaxtyping.jaxtyped` that conditionally applies type checking."""
