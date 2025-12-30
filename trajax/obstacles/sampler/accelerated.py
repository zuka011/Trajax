from typing import cast

from trajax.types import JaxObstacleStateSampler
from trajax.obstacles.accelerated import JaxSampledObstacleStates, JaxObstacleStates

from riskit import distribution

import jax.numpy as jnp

# TODO: Handle issue with random seed.


class JaxGaussianObstacleStateSampler(
    JaxObstacleStateSampler[JaxObstacleStates, JaxSampledObstacleStates]
):
    def __call__[T: int, K: int, N: int](
        self, states: JaxObstacleStates[T, K], *, count: N
    ) -> JaxSampledObstacleStates[T, K, N]:
        if (covariance := states.covariance_array) is None:
            # TODO: Check hos the case where there are no obstacles is handled.
            assert count == 1, (
                "It's pointless to take multiple samples, when covariance information is not available."
            )
            return cast(JaxSampledObstacleStates, states.single())

        T, D_O, _, K = covariance.shape

        mean = jnp.stack([states.x_array, states.y_array, states.heading_array], axis=1)
        flat_covariance = covariance.transpose(0, 3, 1, 2).reshape(-1, D_O, D_O)
        flat_mean = mean.transpose(0, 2, 1).reshape(-1, D_O)

        samples = distribution.jax.gaussian(
            mean=flat_mean, covariance=flat_covariance
        ).sample(count=count)

        samples = samples.reshape(T, K, D_O, count).transpose(0, 2, 1, 3)

        return states.sampled(
            x=samples[:, 0, :, :],
            y=samples[:, 1, :, :],
            heading=samples[:, 2, :, :],  # type: ignore
            sample_count=count,
        )
