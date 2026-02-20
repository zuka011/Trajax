from dataclasses import dataclass

from numtypes import Array, Dims, D

type OriginsArray[V: int] = Array[Dims[V, D[2]]]
type RadiiArray[V: int] = Array[Dims[V]]


@dataclass(frozen=True)
class Circles[V: int]:
    """Describes circles approximating parts of an object for distance computation."""

    origins: OriginsArray[V]
    """The local (x, y) offset of the circle center from the object center."""

    radii: RadiiArray[V]
    """The radius of the circle."""
