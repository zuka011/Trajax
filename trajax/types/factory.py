from typing import Final

from trajax.trajectory.basic import (
    PathParameters as NumPyPathParameters,
    ReferencePoints as NumPyReferencePoints,
    Positions as NumPyPositions,
)
from trajax.trajectory.accelerated import (
    PathParameters as JaxPathParameters,
    ReferencePoints as JaxReferencePoints,
    Positions as JaxPositions,
)
from trajax.types.basic import (
    BasicState as BasicNumPyState,
    BasicStateBatch as BasicNumPyStateBatch,
    BasicControlInputSequence as BasicNumPyControlInputSequence,
    BasicControlInputBatch as BasicNumPyControlInputBatch,
    BasicCosts as BasicNumPyCosts,
)
from trajax.types.accelerated import (
    BasicState as BasicJaxState,
    BasicStateBatch as BasicJaxStateBatch,
    BasicControlInputSequence as BasicJaxControlInputSequence,
    BasicControlInputBatch as BasicJaxControlInputBatch,
    BasicCosts as BasicJaxCosts,
)


class types:
    class numpy:
        type PathParameters[T: int, M: int] = NumPyPathParameters[T, M]
        type ReferencePoints[T: int, M: int] = NumPyReferencePoints[T, M]
        type Positions[T: int, M: int] = NumPyPositions[T, M]

        path_parameters: Final = NumPyPathParameters
        reference_points: Final = NumPyReferencePoints.create
        positions: Final = NumPyPositions.create

        class basic:
            type State[D_x: int] = BasicNumPyState[D_x]
            type StateBatch[T: int, D_x: int, M: int] = BasicNumPyStateBatch[T, D_x, M]
            type ControlInputSequence[T: int, D_u: int] = (
                BasicNumPyControlInputSequence[T, D_u]
            )
            type ControlInputBatch[T: int, D_u: int, M: int] = (
                BasicNumPyControlInputBatch[T, D_u, M]
            )
            type Costs[T: int, M: int] = BasicNumPyCosts[T, M]

            state: Final = BasicNumPyState
            state_batch: Final = BasicNumPyStateBatch
            control_input_sequence: Final = BasicNumPyControlInputSequence
            control_input_batch: Final = BasicNumPyControlInputBatch
            costs: Final = BasicNumPyCosts

    class jax:
        type PathParameters[T: int, M: int] = JaxPathParameters[T, M]
        type ReferencePoints[T: int, M: int] = JaxReferencePoints[T, M]
        type Positions[T: int, M: int] = JaxPositions[T, M]

        path_parameters: Final = JaxPathParameters.create
        reference_points: Final = JaxReferencePoints.create
        positions: Final = JaxPositions.create

        class basic:
            type State[D_x: int] = BasicJaxState[D_x]
            type StateBatch[T: int, D_x: int, M: int] = BasicJaxStateBatch[T, D_x, M]
            type ControlInputSequence[T: int, D_u: int] = BasicJaxControlInputSequence[
                T, D_u
            ]
            type ControlInputBatch[T: int, D_u: int, M: int] = (
                BasicJaxControlInputBatch[T, D_u, M]
            )
            type Costs[T: int, M: int] = BasicJaxCosts[T, M]

            state: Final = BasicJaxState
            state_batch: Final = BasicJaxStateBatch
            control_input_sequence: Final = BasicJaxControlInputSequence
            control_input_batch: Final = BasicJaxControlInputBatch
            costs: Final = BasicJaxCosts
