from dataclasses import dataclass

from trajax import types

from jaxtyping import Array as JaxArray, Float


type NumPyObstacleStates[T: int, K: int] = types.numpy.ObstacleStates[T, K]
type NumPyInitialPositionCovariance[T: int, K: int] = (
    types.numpy.InitialPositionCovariance[K]
)
type NumPyInitialVelocityCovariance[K: int] = types.numpy.InitialVelocityCovariance[K]
type JaxObstacleStates[T: int, K: int] = types.jax.ObstacleStates[T, K]
type JaxInitialPositionCovariance[K: int] = types.jax.InitialPositionCovariance[K]
type JaxInitialVelocityCovariance[K: int] = types.jax.InitialVelocityCovariance[K]


@dataclass(kw_only=True, frozen=True)
class NumPyConstantCovarianceProvider[K: int]:
    position_covariance: NumPyInitialPositionCovariance[K]
    velocity_covariance: NumPyInitialVelocityCovariance[K]

    def position(
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialVelocityCovariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance


@dataclass(kw_only=True, frozen=True)
class JaxConstantCovarianceProvider[K: int]:
    position_covariance: Float[JaxArray, "2 2 K"]
    velocity_covariance: Float[JaxArray, "2 2 K"]

    def position(
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialVelocityCovariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance
