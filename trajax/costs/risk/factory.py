from typing import Final

from trajax.types import NumPyRiskMetric, JaxRiskMetric
from trajax.costs.collision import NoMetric
from trajax.costs.risk.base import RisKitRiskMetric

import riskit as rk


class risk:
    none: Final = NoMetric.create

    class numpy:
        @staticmethod
        def expected_value(*, sample_count: int) -> NumPyRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.numpy.create(),
                creator=lambda *, cost, backend: rk.risk.expected_value_of(
                    cost, backend=backend
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
            )

        @staticmethod
        def mean_variance(*, gamma: float, sample_count: int) -> NumPyRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.numpy.create(),
                creator=lambda *, cost, backend: rk.risk.mean_variance_of(
                    cost, backend=backend, gamma=gamma
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
            )

    class jax:
        @staticmethod
        def expected_value(*, sample_count: int) -> JaxRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.jax.create(),
                creator=lambda *, cost, backend: rk.risk.expected_value_of(
                    cost, backend=backend
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
            )

        @staticmethod
        def mean_variance(*, gamma: float, sample_count: int) -> JaxRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.jax.create(),
                creator=lambda *, cost, backend: rk.risk.mean_variance_of(
                    cost, backend=backend, gamma=gamma
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
            )
