from typing import Sequence, Protocol
from functools import partial

from trajax import ControlInputSequence, risk

from tests.examples import mpcc, reference, obstacles
from tests.benchmarks.runner import (
    run_benchmark,
    BenchmarkRunner,
    NumPyBenchmarkRunner,
    JaxBenchmarkRunner,
)
from tests.benchmarks.mpcc import MpccConfiguration, accumulate_obstacle_states

from pytest import mark
from pytest_benchmark.fixture import BenchmarkFixture

import pytest


class MpccConfigurationProvider(Protocol):
    def __call__(self, sample_count: int) -> MpccConfiguration:
        """Provides an MPCC configuration. The risk metric in the configuration
        will use the given sample count."""
        ...


@mark.slow_benchmark(group="mpcc-risk")
class bench_mpcc_risk:
    @staticmethod
    def cases(runner, risk, mpcc, reference, obstacles) -> Sequence[tuple]:
        return [
            (
                runner.create(),
                partial(
                    lambda sample_count, *, create_risk_metric: MpccConfiguration(
                        planner=(
                            configuration := mpcc.planner_from_mpcc(
                                reference=reference.slalom,
                                obstacles=obstacles.dynamic.slalom,
                                use_covariance_propagation=True,
                                risk_metric=(
                                    risk_metric := create_risk_metric(
                                        sample_count=sample_count
                                    )
                                ),
                            )
                        ).planner,
                        initial_state=configuration.initial_state,
                        nominal_input=configuration.nominal_input,
                        obstacle_simulator=configuration.obstacle_simulator,
                        obstacle_state_observer=configuration.obstacle_state_observer,
                    ),
                    create_risk_metric=create_risk_metric,
                ),
                name,
            )
            for name, create_risk_metric in [
                ("Expected Value", risk.expected_value),
                ("Mean Variance", partial(risk.mean_variance, gamma=0.1)),
                ("VaR", partial(risk.var, alpha=0.9)),
                ("CVaR", partial(risk.cvar, alpha=0.9)),
                ("Entropic Risk", partial(risk.entropic_risk, theta=0.2)),
            ]
        ]

    @mark.parametrize(
        ["runner", "create_configuration", "risk_metric_name"],
        all_cases := [
            *cases(
                runner=NumPyBenchmarkRunner,
                risk=risk.numpy,
                mpcc=mpcc.numpy,
                reference=reference.numpy,
                obstacles=obstacles.numpy,
            ),
            *cases(
                runner=JaxBenchmarkRunner,
                risk=risk.jax,
                mpcc=mpcc.jax,
                reference=reference.jax,
                obstacles=obstacles.jax,
            ),
        ],
        ids=[
            f"{runner.name}-{risk_metric_name}"
            for runner, _, risk_metric_name in all_cases
        ],
    )
    @mark.parametrize(
        "sample_count", [10, 50, 100, 500, 1000], ids=lambda it: f"samples={it}"
    )
    def bench(
        self,
        benchmark: BenchmarkFixture,
        runner: BenchmarkRunner,
        create_configuration: MpccConfigurationProvider,
        risk_metric_name: str,
        sample_count: int,
    ) -> None:
        if runner.is_slow_for(risk_metric_sample_count=sample_count):
            pytest.skip(
                f"Runner {runner.name} is too slow for sample count {sample_count}"
            )

        configuration = create_configuration(sample_count=sample_count)
        planner = configuration.planner
        initial_state = configuration.initial_state
        nominal_input = configuration.nominal_input

        accumulate_obstacle_states(configuration)

        def single_step() -> ControlInputSequence:
            control = planner.step(
                temperature=50,
                nominal_input=nominal_input,
                initial_state=initial_state,
            )
            return control.nominal

        run_benchmark(benchmark, runner, target=single_step)
