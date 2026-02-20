from typing import cast
from dataclasses import dataclass

from faran.types import JaxObstacleStateSampler
from faran.obstacles.accelerated import JaxSampledObstacle2dPoses, JaxObstacle2dPoses

from jaxtyping import PRNGKeyArray
from riskit import distribution

import jax
import jax.numpy as jnp


@dataclass
class JaxGaussianObstacle2dPoseSampler(
    JaxObstacleStateSampler[JaxObstacle2dPoses, JaxSampledObstacle2dPoses]
):
    """Samples obstacle poses from a Gaussian distribution parameterized by predicted covariances."""

    key: PRNGKeyArray

    @staticmethod
    def create(*, seed: int = 42) -> "JaxGaussianObstacle2dPoseSampler":
        return JaxGaussianObstacle2dPoseSampler(key=jax.random.key(seed))

    def __call__[T: int, K: int, N: int](
        self, states: JaxObstacle2dPoses[T, K], *, count: N
    ) -> JaxSampledObstacle2dPoses[T, K, N]:
        if states.count == 0:
            return cast(JaxSampledObstacle2dPoses, states.single())

        if (covariance := states.covariance_array) is None:
            assert count == 1, (
                "It's pointless to take multiple samples, when covariance information is not available."
            )
            return cast(JaxSampledObstacle2dPoses, states.single())

        T, D_O, _, K = covariance.shape

        mean = jnp.stack([states.x_array, states.y_array, states.heading_array], axis=1)
        flat_covariance = covariance.transpose(0, 3, 1, 2).reshape(-1, D_O, D_O)
        flat_mean = mean.transpose(0, 2, 1).reshape(-1, D_O)

        gaussian = distribution.jax.gaussian(
            mean=flat_mean, covariance=flat_covariance, key=self.key
        )
        samples = gaussian.sample(count=count)
        self.key = gaussian.key

        samples = samples.reshape(T, K, D_O, count).transpose(0, 2, 1, 3)

        return states.sampled(
            x=samples[:, 0, :, :],
            y=samples[:, 1, :, :],
            heading=samples[:, 2, :, :],  # type: ignore
            sample_count=count,
        )
