from typing import Final

from faran.types import NumPyRisk, NumPyRiskMetric, JaxRisk, JaxRiskMetric
from faran.costs.collision import NumPyNoMetric, JaxNoMetric
from faran.costs.risk.base import RisKitRiskMetric

import riskit as rk


class risk:
    """Factory namespace for creating risk metrics with different risk measures."""

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

        @staticmethod
        def var(*, alpha: float, sample_count: int) -> NumPyRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.numpy.create(),
                creator=lambda *, cost, backend: rk.risk.var_of(
                    cost, backend=backend, alpha=alpha
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=NumPyRisk,
                name="Value at Risk",
            )

        @staticmethod
        def cvar(*, alpha: float, sample_count: int) -> NumPyRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.numpy.create(),
                creator=lambda *, cost, backend: rk.risk.cvar_of(
                    cost, backend=backend, alpha=alpha
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=NumPyRisk,
                name="Conditional Value at Risk",
            )

        @staticmethod
        def entropic_risk(*, theta: float, sample_count: int) -> NumPyRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.numpy.create(),
                creator=lambda *, cost, backend: rk.risk.entropic_risk_of(
                    cost, backend=backend, theta=theta
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=NumPyRisk,
                name="Entropic Risk",
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

        @staticmethod
        def var(*, alpha: float, sample_count: int) -> JaxRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.jax.create(),
                creator=lambda *, cost, backend: rk.risk.var_of(
                    cost, backend=backend, alpha=alpha
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=JaxRisk,
                name="Value at Risk",
            )

        @staticmethod
        def cvar(*, alpha: float, sample_count: int) -> JaxRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.jax.create(),
                creator=lambda *, cost, backend: rk.risk.cvar_of(
                    cost, backend=backend, alpha=alpha
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=JaxRisk,
                name="Conditional Value at Risk",
            )

        @staticmethod
        def entropic_risk(*, theta: float, sample_count: int) -> JaxRiskMetric:
            return RisKitRiskMetric.create(
                backend=rk.backend.jax.create(),
                creator=lambda *, cost, backend: rk.risk.entropic_risk_of(
                    cost, backend=backend, theta=theta
                ).sampled_with(rk.sampler.monte_carlo(sample_count)),
                to_risk=JaxRisk,
                name="Entropic Risk",
            )
