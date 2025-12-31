from typing import cast
from dataclasses import dataclass

from trajax.types import JaxObstacleStateSampler
from trajax.obstacles.accelerated import JaxSampledObstacleStates, JaxObstacleStates

from jaxtyping import PRNGKeyArray
from riskit import distribution

import jax
import jax.numpy as jnp


@dataclass
class JaxGaussianObstacleStateSampler(
    JaxObstacleStateSampler[JaxObstacleStates, JaxSampledObstacleStates]
):
    key: PRNGKeyArray

    @staticmethod
    def create(*, seed: int = 42) -> "JaxGaussianObstacleStateSampler":
        return JaxGaussianObstacleStateSampler(key=jax.random.key(seed))

    def __call__[T: int, K: int, N: int](
        self, states: JaxObstacleStates[T, K], *, count: N
    ) -> JaxSampledObstacleStates[T, K, N]:
        if states.count == 0:
            return cast(JaxSampledObstacleStates, states.single())

        if (covariance := states.covariance_array) is None:
            assert count == 1, (
                "It's pointless to take multiple samples, when covariance information is not available."
            )
            return cast(JaxSampledObstacleStates, states.single())

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
