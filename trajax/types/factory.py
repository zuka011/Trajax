from typing import Final, Any

from trajax.mppi import (
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPyCostFunction,
    JaxState,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxCosts,
    JaxCostFunction,
    State as State,
    StateBatch as StateBatch,
    ControlInputSequence as ControlInputSequence,
    ControlInputBatch as ControlInputBatch,
    Costs as Costs,
    CostFunction as CostFunction,
)
from trajax.models import (
    BICYCLE_D_X,
    BICYCLE_D_U,
    BicycleD_x,
    BicycleD_u,
    BicycleState,
    BicycleStateBatch,
    BicyclePositions,
    BicycleControlInputSequence,
    BicycleControlInputBatch,
    NumPyBicycleState,
    NumPyBicycleStateBatch,
    NumPyBicyclePositions,
    NumPyBicycleControlInputSequence,
    NumPyBicycleControlInputBatch,
    JaxBicycleState,
    JaxBicycleStateBatch,
    JaxBicyclePositions,
    JaxBicycleControlInputSequence,
    JaxBicycleControlInputBatch,
)
from trajax.trajectory import (
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
)
from trajax.costs import (
    NumPyPathParameterExtractor,
    NumPyPathVelocityExtractor,
    NumPyPositionExtractor,
    NumPyContouringCost,
    JaxPathParameterExtractor,
    JaxPathVelocityExtractor,
    JaxPositionExtractor,
    JaxContouringCost,
    Error as Error,
    ContouringCost as ContouringCost,
)
from trajax.states import (
    NumPySimpleState,
    NumPySimpleStateBatch,
    NumPySimpleControlInputSequence,
    NumPySimpleControlInputBatch,
    NumPySimpleCosts,
    JaxSimpleState,
    JaxSimpleStateBatch,
    JaxSimpleControlInputSequence,
    JaxSimpleControlInputBatch,
    JaxSimpleCosts,
    NumPyAugmentedState,
    NumPyAugmentedStateBatch,
    NumPyAugmentedControlInputSequence,
    NumPyAugmentedControlInputBatch,
    JaxAugmentedState,
    JaxAugmentedStateBatch,
    JaxAugmentedControlInputSequence,
    JaxAugmentedControlInputBatch,
    AugmentedState,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
)


class types:
    type State[D_x: int = Any] = State[D_x]
    type StateBatch[T: int = Any, D_x: int = Any, M: int = Any] = StateBatch[T, D_x, M]
    type ControlInputSequence[T: int = Any, D_u: int = Any] = ControlInputSequence[
        T, D_u
    ]
    type ControlInputBatch[T: int = Any, D_u: int = Any, M: int = Any] = (
        ControlInputBatch[T, D_u, M]
    )
    type Costs[T: int = Any, M: int = Any] = Costs[T, M]
    type CostFunction[I: ControlInputBatch, S: StateBatch, C: Costs] = CostFunction[
        I, S, C
    ]
    type Error[T: int = Any, M: int = Any] = Error[T, M]
    type ContouringCost[I: ControlInputBatch, S: StateBatch, C: Costs, D: Error] = (
        ContouringCost[I, S, C, D]
    )

    class bicycle:
        type D_x = BicycleD_x
        type D_u = BicycleD_u
        type State = BicycleState
        type StateBatch[T: int = Any, M: int = Any] = BicycleStateBatch[T, M]
        type Positions[T: int = Any, M: int = Any] = BicyclePositions[T, M]
        type ControlInputSequence[T: int = Any] = BicycleControlInputSequence[T]
        type ControlInputBatch[T: int = Any, M: int = Any] = BicycleControlInputBatch[
            T, M
        ]

        D_X: Final = BICYCLE_D_X
        D_U: Final = BICYCLE_D_U

    class augmented:
        type State[P: State, V: State] = AugmentedState[P, V]
        type StateBatch[P: StateBatch, V: StateBatch] = AugmentedStateBatch[P, V]
        type ControlInputSequence[P: ControlInputSequence, V: ControlInputSequence] = (
            AugmentedControlInputSequence[P, V]
        )
        type ControlInputBatch[P: ControlInputBatch, V: ControlInputBatch] = (
            AugmentedControlInputBatch[P, V]
        )

        state: Final = AugmentedState
        state_batch: Final = AugmentedStateBatch
        control_input_sequence: Final = AugmentedControlInputSequence
        control_input_batch: Final = AugmentedControlInputBatch

    class numpy:
        type State[D_x: int = Any] = NumPyState[D_x]
        type StateBatch[T: int = Any, D_x: int = Any, M: int = Any] = NumPyStateBatch[
            T, D_x, M
        ]
        type ControlInputSequence[T: int = Any, D_u: int = Any] = (
            NumPyControlInputSequence[T, D_u]
        )
        type ControlInputBatch[T: int = Any, D_u: int = Any, M: int = Any] = (
            NumPyControlInputBatch[T, D_u, M]
        )
        type Costs[T: int = Any, M: int = Any] = NumPyCosts[T, M]
        type PathParameters[T: int = Any, M: int = Any] = NumPyPathParameters[T, M]
        type ReferencePoints[T: int = Any, M: int = Any] = NumPyReferencePoints[T, M]
        type Positions[T: int = Any, M: int = Any] = NumPyPositions[T, M]

        type CostFunction[
            T: int = Any,
            D_u: int = Any,
            D_x: int = Any,
            M: int = Any,
        ] = NumPyCostFunction[
            NumPyControlInputBatch[T, D_u, M],
            NumPyStateBatch[T, D_x, M],
            NumPyCosts[T, M],
        ]
        type PathParameterExtractor[S: NumPyStateBatch] = NumPyPathParameterExtractor[S]
        type PathVelocityExtractor[I: NumPyControlInputBatch] = (
            NumPyPathVelocityExtractor[I]
        )
        type PositionExtractor[S: NumPyStateBatch] = NumPyPositionExtractor[S]
        type ContouringCost[S: NumPyStateBatch] = NumPyContouringCost[S]

        path_parameters: Final = NumPyPathParameters
        reference_points: Final = NumPyReferencePoints.create
        positions: Final = NumPyPositions.create

        class simple:
            type State[D_x: int = Any] = NumPySimpleState[D_x]
            type StateBatch[T: int = Any, D_x: int = Any, M: int = Any] = (
                NumPySimpleStateBatch[T, D_x, M]
            )
            type ControlInputSequence[T: int = Any, D_u: int = Any] = (
                NumPySimpleControlInputSequence[T, D_u]
            )
            type ControlInputBatch[T: int = Any, D_u: int = Any, M: int = Any] = (
                NumPySimpleControlInputBatch[T, D_u, M]
            )
            type Costs[T: int = Any, M: int = Any] = NumPySimpleCosts[T, M]

            state: Final = NumPySimpleState
            state_batch: Final = NumPySimpleStateBatch
            control_input_sequence: Final = NumPySimpleControlInputSequence
            control_input_batch: Final = NumPySimpleControlInputBatch
            costs: Final = NumPySimpleCosts

        class bicycle:
            type State = NumPyBicycleState
            type StateBatch[T: int = Any, M: int = Any] = NumPyBicycleStateBatch[T, M]
            type Positions[T: int = Any, M: int = Any] = NumPyBicyclePositions[T, M]
            type ControlInputSequence[T: int = Any] = NumPyBicycleControlInputSequence[
                T
            ]
            type ControlInputBatch[T: int = Any, M: int = Any] = (
                NumPyBicycleControlInputBatch[T, M]
            )

            state: Final = NumPyBicycleState.create
            state_batch: Final = NumPyBicycleStateBatch
            positions: Final = NumPyBicyclePositions
            control_input_sequence: Final = NumPyBicycleControlInputSequence
            control_input_batch: Final = NumPyBicycleControlInputBatch

        class augmented:
            type State[P: NumPyState, V: NumPyState] = NumPyAugmentedState[P, V]
            type StateBatch[P: NumPyStateBatch, V: NumPyStateBatch] = (
                NumPyAugmentedStateBatch[P, V]
            )
            type ControlInputSequence[
                P: NumPyControlInputSequence,
                V: NumPyControlInputSequence,
            ] = NumPyAugmentedControlInputSequence[P, V]
            type ControlInputBatch[
                P: NumPyControlInputBatch,
                V: NumPyControlInputBatch,
            ] = NumPyAugmentedControlInputBatch[P, V]

            state: Final = NumPyAugmentedState
            state_batch: Final = NumPyAugmentedStateBatch
            control_input_sequence: Final = NumPyAugmentedControlInputSequence
            control_input_batch: Final = NumPyAugmentedControlInputBatch

    class jax:
        type State[D_x: int = Any] = JaxState[D_x]
        type StateBatch[T: int = Any, D_x: int = Any, M: int = Any] = JaxStateBatch[
            T, D_x, M
        ]
        type ControlInputSequence[T: int = Any, D_u: int = Any] = (
            JaxControlInputSequence[T, D_u]
        )
        type ControlInputBatch[T: int = Any, D_u: int = Any, M: int = Any] = (
            JaxControlInputBatch[T, D_u, M]
        )
        type Costs[T: int = Any, M: int = Any] = JaxCosts[T, M]
        type PathParameters[T: int = Any, M: int = Any] = JaxPathParameters[T, M]
        type ReferencePoints[T: int = Any, M: int = Any] = JaxReferencePoints[T, M]
        type Positions[T: int = Any, M: int = Any] = JaxPositions[T, M]

        type CostFunction[
            T: int = Any,
            D_u: int = Any,
            D_x: int = Any,
            M: int = Any,
        ] = JaxCostFunction[
            JaxControlInputBatch[T, D_u, M], JaxStateBatch[T, D_x, M], JaxCosts[T, M]
        ]
        type PathParameterExtractor[S: JaxStateBatch] = JaxPathParameterExtractor[S]
        type PathVelocityExtractor[I: JaxControlInputBatch] = JaxPathVelocityExtractor[
            I
        ]
        type PositionExtractor[S: JaxStateBatch] = JaxPositionExtractor[S]
        type ContouringCost[S: JaxStateBatch] = JaxContouringCost[S]

        path_parameters: Final = JaxPathParameters.create
        reference_points: Final = JaxReferencePoints.create
        positions: Final = JaxPositions.create

        class simple:
            type State[D_x: int = Any] = JaxSimpleState[D_x]
            type StateBatch[T: int = Any, D_x: int = Any, M: int = Any] = (
                JaxSimpleStateBatch[T, D_x, M]
            )
            type ControlInputSequence[T: int = Any, D_u: int = Any] = (
                JaxSimpleControlInputSequence[T, D_u]
            )
            type ControlInputBatch[T: int = Any, D_u: int = Any, M: int = Any] = (
                JaxSimpleControlInputBatch[T, D_u, M]
            )
            type Costs[T: int = Any, M: int = Any] = JaxSimpleCosts[T, M]

            state: Final = JaxSimpleState
            state_batch: Final = JaxSimpleStateBatch
            control_input_sequence: Final = JaxSimpleControlInputSequence
            control_input_batch: Final = JaxSimpleControlInputBatch
            costs: Final = JaxSimpleCosts

        class bicycle:
            type State = JaxBicycleState
            type StateBatch[T: int = Any, M: int = Any] = JaxBicycleStateBatch[T, M]
            type Positions[T: int = Any, M: int = Any] = JaxBicyclePositions[T, M]
            type ControlInputSequence[T: int = Any] = JaxBicycleControlInputSequence[T]
            type ControlInputBatch[T: int = Any, M: int = Any] = (
                JaxBicycleControlInputBatch[T, M]
            )

            state: Final = JaxBicycleState
            state_batch: Final = JaxBicycleStateBatch
            positions: Final = JaxBicyclePositions
            control_input_sequence: Final = JaxBicycleControlInputSequence
            control_input_batch: Final = JaxBicycleControlInputBatch

        class augmented:
            type State[P: JaxState, V: JaxState] = JaxAugmentedState[P, V]
            type StateBatch[P: JaxStateBatch, V: JaxStateBatch] = (
                JaxAugmentedStateBatch[P, V]
            )
            type ControlInputSequence[
                P: JaxControlInputSequence,
                V: JaxControlInputSequence,
            ] = JaxAugmentedControlInputSequence[P, V]
            type ControlInputBatch[P: JaxControlInputBatch, V: JaxControlInputBatch] = (
                JaxAugmentedControlInputBatch[P, V]
            )

            state: Final = JaxAugmentedState
            state_batch: Final = JaxAugmentedStateBatch
            control_input_sequence: Final = JaxAugmentedControlInputSequence
            control_input_batch: Final = JaxAugmentedControlInputBatch
