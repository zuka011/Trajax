from dataclasses import dataclass

from trajax import State, ControlInputSequence, Mppi

from tests.benchmarks.runner import (
    run_benchmark,
    BenchmarkRunner,
    NumPyBenchmarkRunner,
    JaxBenchmarkRunner,
)
from tests.examples import numpy_mpcc, jax_mpcc

from pytest import mark
from pytest_benchmark.fixture import BenchmarkFixture


@dataclass(frozen=True)
class MpccConfiguration[StateT: State, InputT: ControlInputSequence]:
    planner: Mppi[StateT, InputT]
    initial_state: StateT
    nominal_input: InputT


@mark.parametrize(
    ["id", "runner", "configuration"],
    [
        *[
            (
                id,
                NumPyBenchmarkRunner.create(),
                MpccConfiguration(
                    planner=configuration.planner,
                    initial_state=configuration.initial_state,
                    nominal_input=configuration.nominal_input,
                ),
            )
            for id, configuration in (
                ("NumPy", numpy_mpcc.configure.numpy_mpcc_planner_from_augmented()),
            )
        ],
        *[
            (
                id,
                JaxBenchmarkRunner.create(),
                MpccConfiguration(
                    planner=configuration.planner,
                    initial_state=configuration.initial_state,
                    nominal_input=configuration.nominal_input,
                ),
            )
            for id, configuration in (
                ("JAX", jax_mpcc.configure.jax_mpcc_planner_from_augmented()),
            )
        ],
    ],
)
@mark.benchmark(group="mpcc-single-step")
def bench_mpcc_single_step(
    benchmark: BenchmarkFixture,
    id: str,
    runner: BenchmarkRunner,
    configuration: MpccConfiguration,
) -> None:
    planner = configuration.planner
    initial_state = configuration.initial_state
    nominal_input = configuration.nominal_input

    async def single_step() -> ControlInputSequence:
        control = await planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        return control.nominal

    run_benchmark(benchmark, runner, target=single_step)
