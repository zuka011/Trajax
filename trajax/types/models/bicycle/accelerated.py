from typing import Protocol

from trajax.types.predictors import ObstacleStatesHistory
from trajax.types.models.bicycle.common import BicycleD_o

from jaxtyping import Array as JaxArray, Float


class JaxBicycleObstacleStatesHistory[T: int, K: int](
    ObstacleStatesHistory[T, BicycleD_o, K], Protocol
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
