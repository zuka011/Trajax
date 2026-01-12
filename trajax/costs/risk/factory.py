from typing import Final

from trajax.types import NumPyRisk, NumPyRiskMetric, JaxRisk, JaxRiskMetric
from trajax.costs.collision import NumPyNoMetric, JaxNoMetric
from trajax.costs.risk.base import RisKitRiskMetric

import riskit as rk


class risk:
    class numpy:
        none: Final = NumPyNoMetric.create

        @staticmethod
        def expected_value(*, sample_count: int) -> NumPyRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.numpy.create(),
                creator=lambda *, cost, backend: rk.risk.expected_value_of(
                    cost, backend=backend
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=NumPyRisk,
                name="Expected Value",
            )

        @staticmethod
        def mean_variance(*, gamma: float, sample_count: int) -> NumPyRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.numpy.create(),
                creator=lambda *, cost, backend: rk.risk.mean_variance_of(
                    cost, backend=backend, gamma=gamma
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=NumPyRisk,
                name="Mean-Variance",
            )

    class jax:
        none: Final = JaxNoMetric.create

        @staticmethod
        def expected_value(*, sample_count: int) -> JaxRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.jax.create(),
                creator=lambda *, cost, backend: rk.risk.expected_value_of(
                    cost, backend=backend
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=JaxRisk,
                name="Expected Value",
            )

        @staticmethod
        def mean_variance(*, gamma: float, sample_count: int) -> JaxRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.jax.create(),
                creator=lambda *, cost, backend: rk.risk.mean_variance_of(
                    cost, backend=backend, gamma=gamma
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=JaxRisk,
                name="Mean-Variance",
            )
