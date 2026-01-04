from typing import Protocol

from trajax.types.predictors.common import ObstacleStatesHistory

from jaxtyping import Float, Array as JaxArray


class JaxObstacleStatesHistory[T: int, D_o: int, K: int](
    ObstacleStatesHistory[T, D_o, K], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle state history as a JAX array."""
        ...
