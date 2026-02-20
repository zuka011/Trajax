from faran import ControlInputSequence

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


@mark.parametrize(
    ["runner", "configuration"],
    [
        (
            NumPyBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(np_configuration := mpcc.numpy.planner_from_mpcc()).planner,
                initial_state=np_configuration.initial_state,
                nominal_input=np_configuration.nominal_input,
            ),
        ),
        (
            JaxBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(jax_configuration := mpcc.jax.planner_from_mpcc()).planner,
                initial_state=jax_configuration.initial_state,
                nominal_input=jax_configuration.nominal_input,
            ),
        ),
    ],
    ids=["NumPy", "JAX"],
)
@mark.benchmark(group="mpcc-single-step")
def bench_mpcc_single_step(
    benchmark: BenchmarkFixture,
    runner: BenchmarkRunner,
    configuration: MpccConfiguration,
) -> None:
    planner = configuration.planner
    initial_state = configuration.initial_state
    nominal_input = configuration.nominal_input

    def single_step() -> ControlInputSequence:
        control = planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        return control.nominal

    run_benchmark(benchmark, runner, target=single_step)


@mark.parametrize(
    ["runner", "configuration"],
    [
        (
            NumPyBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(
                    np_configuration := mpcc.numpy.planner_from_mpcc(
                        reference=reference.numpy.small_circle,
                        obstacles=obstacles.numpy.static.loop,
                    )
                ).planner,
                initial_state=np_configuration.initial_state,
                nominal_input=np_configuration.nominal_input,
                obstacle_simulator=np_configuration.obstacle_simulator,
                obstacle_state_observer=np_configuration.obstacle_state_observer,
            ),
        ),
        (
            JaxBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(
                    jax_configuration := mpcc.jax.planner_from_mpcc(
                        reference=reference.jax.small_circle,
                        obstacles=obstacles.jax.static.loop,
                    )
                ).planner,
                initial_state=jax_configuration.initial_state,
                nominal_input=jax_configuration.nominal_input,
                obstacle_simulator=jax_configuration.obstacle_simulator,
                obstacle_state_observer=jax_configuration.obstacle_state_observer,
            ),
        ),
    ],
    ids=["NumPy", "JAX"],
)
@mark.benchmark(group="mpcc-static-obstacles-single-step")
def bench_mpcc_static_obstacles_single_step(
    benchmark: BenchmarkFixture,
    runner: BenchmarkRunner,
    configuration: MpccConfiguration,
) -> None:
    planner = configuration.planner
    initial_state = configuration.initial_state
    nominal_input = configuration.nominal_input

    accumulate_obstacle_states(configuration)

    def single_step() -> ControlInputSequence:
        control = planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        return control.nominal

    run_benchmark(benchmark, runner, target=single_step)


@mark.parametrize(
    ["runner", "configuration"],
    [
        (
            NumPyBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(
                    np_configuration := mpcc.numpy.planner_from_mpcc(
                        reference=reference.numpy.slalom,
                        obstacles=obstacles.numpy.dynamic.slalom,
                    )
                ).planner,
                initial_state=np_configuration.initial_state,
                nominal_input=np_configuration.nominal_input,
                obstacle_simulator=np_configuration.obstacle_simulator,
                obstacle_state_observer=np_configuration.obstacle_state_observer,
            ),
        ),
        (
            JaxBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(
                    jax_configuration := mpcc.jax.planner_from_mpcc(
                        reference=reference.jax.slalom,
                        obstacles=obstacles.jax.dynamic.slalom,
                    )
                ).planner,
                initial_state=jax_configuration.initial_state,
                nominal_input=jax_configuration.nominal_input,
                obstacle_simulator=jax_configuration.obstacle_simulator,
                obstacle_state_observer=jax_configuration.obstacle_state_observer,
            ),
        ),
    ],
    ids=["NumPy", "JAX"],
)
@mark.benchmark(group="mpcc-dynamic-obstacles-single-step")
def bench_mpcc_dynamic_obstacles_single_step(
    benchmark: BenchmarkFixture,
    runner: BenchmarkRunner,
    configuration: MpccConfiguration,
) -> None:
    planner = configuration.planner
    initial_state = configuration.initial_state
    nominal_input = configuration.nominal_input

    accumulate_obstacle_states(configuration)

    def single_step() -> ControlInputSequence:
        control = planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        return control.nominal

    run_benchmark(benchmark, runner, target=single_step)


@mark.parametrize(
    ["runner", "configuration"],
    [
        (
            NumPyBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(
                    np_configuration := mpcc.numpy.planner_from_mpcc(
                        reference=reference.numpy.slalom,
                        obstacles=obstacles.numpy.dynamic.slalom,
                        use_risk_metric=True,
                    )
                ).planner,
                initial_state=np_configuration.initial_state,
                nominal_input=np_configuration.nominal_input,
                obstacle_simulator=np_configuration.obstacle_simulator,
                obstacle_state_observer=np_configuration.obstacle_state_observer,
            ),
        ),
        (
            JaxBenchmarkRunner.create(),
            MpccConfiguration(
                planner=(
                    jax_configuration := mpcc.jax.planner_from_mpcc(
                        reference=reference.jax.slalom,
                        obstacles=obstacles.jax.dynamic.slalom,
                        use_risk_metric=True,
                    )
                ).planner,
                initial_state=jax_configuration.initial_state,
                nominal_input=jax_configuration.nominal_input,
                obstacle_simulator=jax_configuration.obstacle_simulator,
                obstacle_state_observer=jax_configuration.obstacle_state_observer,
            ),
        ),
    ],
    ids=["NumPy", "JAX"],
)
@mark.benchmark(group="mpcc-dynamic-uncertain-obstacles-single-step")
def bench_mpcc_dynamic_uncertain_obstacles_single_step(
    benchmark: BenchmarkFixture,
    runner: BenchmarkRunner,
    configuration: MpccConfiguration,
) -> None:
    planner = configuration.planner
    initial_state = configuration.initial_state
    nominal_input = configuration.nominal_input

    accumulate_obstacle_states(configuration)

    def single_step() -> ControlInputSequence:
        control = planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        return control.nominal

    run_benchmark(benchmark, runner, target=single_step)
