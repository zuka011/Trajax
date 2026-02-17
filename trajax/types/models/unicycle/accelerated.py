from typing import Protocol

from trajax.types.predictors import ObstacleStatesHistory
from trajax.types.models.unicycle.common import UnicycleD_o, UNICYCLE_D_O

from jaxtyping import Array as JaxArray, Float


class JaxUnicycleObstacleStatesHistory[T: int, K: int](
    ObstacleStatesHistory[T, UnicycleD_o, K], Protocol
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
    def array(self) -> Float[JaxArray, f"T {UNICYCLE_D_O} K"]:
        """Returns the obstacle history as a JAX array."""
        ...
