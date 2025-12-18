from typing import Protocol, Sequence

from trajax import (
    AugmentedState,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
    Trajectory,
    ContouringCost,
    Mppi,
)

import numpy as np
from numtypes import Array, Dim1

from tests.visualize import VisualizationData, visualizer, MpccSimulationResult
from tests.examples import numpy_mpcc
from pytest import mark


class StateStacker[StateT, StateBatchT](Protocol):
    def __call__(self, states: Sequence[StateT]) -> StateBatchT:
        """Stacks a sequence of states into a state batch."""
        ...


class ZeroControlInputProvider[ControlInputBatchT](Protocol):
    def __call__(self, horizon: int) -> ControlInputBatchT:
        """Returns a control input batch of zeroes with the specified horizon."""
        ...


class MpccPlannerConfiguration[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](Protocol):
    @property
    def reference(self) -> Trajectory:
        """Returns the reference trajectory."""
        ...

    @property
    def planner(self) -> Mppi[StateT, ControlInputSequenceT]:
        """Returns the MPCC planner."""
        ...

    @property
    def model(
        self,
    ) -> DynamicalModel[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT]:
        """Returns the augmented dynamical model."""
        ...

    @property
    def contouring_cost(self) -> ContouringCost[ControlInputBatchT, StateBatchT]:
        """Returns the contouring cost function."""
        ...

    @property
    def initial_state(self) -> StateT:
        """Returns the initial augmented state."""
        ...

    @property
    def nominal_input(self) -> ControlInputSequenceT:
        """Returns the nominal control input sequence."""
        ...

    @property
    def wheelbase(self) -> float:
        """Returns the vehicle wheelbase (to be used in the visualization)."""
        ...

    @property
    def stack_states(self) -> StateStacker[StateT, StateBatchT]:
        """Returns the state stacker."""
        ...

    @property
    def zero_inputs(self) -> ZeroControlInputProvider[ControlInputBatchT]:
        """Returns the control input provider."""
        ...


@mark.asyncio
@mark.parametrize(
    ["configuration", "configuration_name"],
    [
        (numpy_mpcc.configure.numpy_mpcc_planner_from_base(), "numpy-mpcc-from-base"),
        (
            numpy_mpcc.configure.numpy_mpcc_planner_from_augmented(),
            "numpy-mpcc-from-augmented",
        ),
    ],
)
@mark.visualize.with_args(visualizer.mpcc(), lambda seed: f"mpcc-{seed}")
async def test_that_mpcc_planner_follows_trajectory_without_excessive_deviation[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](
    visualization: VisualizationData[MpccSimulationResult],
    configuration: MpccPlannerConfiguration[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    configuration_name: str,
) -> None:
    reference = configuration.reference
    planner = configuration.planner
    augmented_model = configuration.model
    contouring_cost = configuration.contouring_cost
    current_state = configuration.initial_state
    nominal_input = configuration.nominal_input
    L = configuration.wheelbase

    states: list[StateT] = []
    min_progress = reference.path_length * 0.7
    progress = 0.0

    for _ in range(max_steps := 300):
        control = await planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=current_state,
        )

        nominal_input = control.nominal

        states.append(
            current_state := await augmented_model.step(
                input=control.optimal, state=current_state
            )
        )

        if (progress := current_state.virtual.array[0]) >= min_progress:
            break

    visualization.data_is(
        MpccSimulationResult(
            reference=reference,
            states=states,
            contouring_errors=(
                errors := contouring_errors(
                    contouring_cost,
                    inputs=configuration.zero_inputs(horizon=len(states)),
                    states=configuration.stack_states(states),
                )
            ),
            wheelbase=L,
        )
    ).seed_is(configuration_name)

    assert progress > min_progress, (
        f"Vehicle did not make sufficient progress along the path in {max_steps} steps. "
        f"Final path parameter: {progress:.1f}, expected > {min_progress:.1f}"
    )

    assert (deviation := errors.max()) < (max_deviation := 1.0), (
        f"Vehicle deviated too far from the reference trajectory. "
        f"Max lateral deviation: {deviation:.2f} m, expected < {max_deviation:.2f} m"
    )


def contouring_errors[InputT: ControlInputBatch, StateT: StateBatch](
    contouring: ContouringCost[InputT, StateT], inputs: InputT, states: StateT
) -> Array[Dim1]:
    return np.asarray(contouring.error(inputs=inputs, states=states))[:, 0]
