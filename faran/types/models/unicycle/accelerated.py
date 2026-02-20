from typing import Protocol, Any

from faran.types.predictors import ObstacleStatesHistory

from jaxtyping import Array as JaxArray, Float


class JaxUnicycleObstacleStatesHistory[T: int, K: int](
    ObstacleStatesHistory[T, Any, K], Protocol
):
    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        """Returns the x positions of the obstacles over time."""
        ...

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        """Returns the y positions of the obstacles over time."""
        ...

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        """Returns the headings of the obstacles over time."""
        ...

    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle history as a JAX array."""
        ...
