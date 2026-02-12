from typing import Protocol, Any

from trajax.types.predictors.common import (
    ObstacleStatesHistory,
    ObstacleStateSequences,
    ObstacleControlInputSequences,
)

from jaxtyping import Float, Array as JaxArray


class JaxObstacleStatesHistory[T: int, D_o: int, K: int, ObstacleStatesForTimeStepT](
    ObstacleStatesHistory[T, D_o, K, ObstacleStatesForTimeStepT], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle state history as a JAX array."""
        ...


class JaxObstacleStateSequences[T: int, D_o: int, K: int, SingleSampleT = Any](
    ObstacleStateSequences[T, D_o, K, SingleSampleT], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle state sequences as a JAX array."""
        ...


class JaxObstacleControlInputSequences[T: int, D_u: int, K: int](
    ObstacleControlInputSequences[T, D_u, K], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u K"]:
        """Returns the obstacle control input sequences as a JAX array."""
        ...
