from dataclasses import dataclass

from numtypes import Array, Dims, D, array

import numpy as np

type VerticesArray[P: int = int] = Array[Dims[P, D[2]]]


@dataclass(frozen=True)
class ConvexPolygon:
    """Describes a convex polygon for SAT-based distance computation.

    Vertices must be in counter-clockwise order in the local frame.
    """

    vertices: VerticesArray
    """The (x, y) coordinates of the polygon vertices in local frame, CCW order."""

    def __post_init__(self) -> None:
        verify_positive_edge_lengths(self.vertices)

    @staticmethod
    def square(*, size: float = 1.0) -> "ConvexPolygon":
        """A square centered at the origin, CCW order.

        Args:
            size: The length of each side of the square.
        """
        return ConvexPolygon(
            array(
                [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                shape=(4, 2),
            )
            * size
        )

    @staticmethod
    def rectangle(*, length: float = 1.0, width: float = 1.0) -> "ConvexPolygon":
        """Rectangle centered at origin, CCW order. Length along x, width along y.

        Args:
            length: The length of the rectangle along the x-axis.
            width: The width of the rectangle along the y-axis.
        """
        half_length = length / 2
        half_width = width / 2
        return ConvexPolygon(
            array(
                [
                    [-half_length, -half_width],
                    [half_length, -half_width],
                    [half_length, half_width],
                    [-half_length, half_width],
                ],
                shape=(4, 2),
            )
        )


def verify_positive_edge_lengths(vertices: VerticesArray) -> None:
    edges = np.roll(vertices, -1, axis=0) - vertices
    edge_lengths = np.linalg.norm(edges, axis=-1)

    assert np.all(edge_lengths > 1e-10), (
        f"Polygon edges must have non-zero length. Got lengths: {edge_lengths}"
    )
