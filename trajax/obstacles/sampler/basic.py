from typing import cast

from trajax.costs import (
    NumPyObstacleStates,
    NumPySampledObstacleStates,
    NumPyObstacleStateSampler,
)

from numtypes import shape_of
from riskit import distribution

import numpy as np


# TODO: Handle issue with random seed.


class NumPyGaussianObstacleStateSampler(
    NumPyObstacleStateSampler[NumPyObstacleStates, NumPySampledObstacleStates]
):
    def __call__[T: int, K: int, N: int](
        self, states: NumPyObstacleStates[T, K], *, count: N
    ) -> NumPySampledObstacleStates[T, K, N]:
        if (covariance := states.covariance()) is None:
            assert count == 1, (
                "It's pointless to take multiple samples, when covariance information is not available."
            )
            return cast(NumPySampledObstacleStates, states.single())

        T, D_O, _, K = covariance.shape

        mean = np.stack([states.x(), states.y(), states.heading()], axis=1)
        flat_covariance = covariance.transpose(0, 3, 1, 2).reshape(-1, D_O, D_O)
        flat_mean = mean.transpose(0, 2, 1).reshape(-1, D_O)

        assert shape_of(flat_covariance, matches=(T * K, D_O, D_O), name="covariance")
        assert shape_of(flat_mean, matches=(T * K, D_O), name="mean")

        samples = distribution.numpy.gaussian(
            mean=flat_mean, covariance=flat_covariance
        ).sample(count=count)

        assert shape_of(samples, matches=(T * K, D_O, count), name="samples")

        samples = samples.reshape(T, K, D_O, count).transpose(0, 2, 1, 3)

        assert shape_of(samples, matches=(T, D_O, K, count), name="samples")

        return states.sampled(
            x=samples[:, 0, :, :], y=samples[:, 1, :, :], heading=samples[:, 2, :, :]
        )
