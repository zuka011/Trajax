from dataclasses import dataclass


from numtypes import Array, Dims, D

type OriginsArray[N: int] = Array[Dims[N, D[2]]]
type RadiiArray[N: int] = Array[Dims[N]]


@dataclass(frozen=True)
class Circles[N: int]:
    """Describes circles approximating parts of an object for distance computation."""

    origins: OriginsArray[N]
    """The local (x, y) offset of the circle center from the object center."""

    radii: RadiiArray[N]
    """The radius of the circle."""
