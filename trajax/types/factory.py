from typing import Final

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
    AugmentedState,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
)


class types:
    type State[D_x: int = int] = State[D_x]
    type StateBatch[T: int = int, D_x: int = int, M: int = int] = StateBatch[T, D_x, M]
    type ControlInputSequence[T: int = int, D_u: int = int] = ControlInputSequence[
        T, D_u
    ]
    type ControlInputBatch[T: int = int, D_u: int = int, M: int = int] = (
        ControlInputBatch[T, D_u, M]
    )
    type Costs[T: int = int, M: int = int] = Costs[T, M]
    type CostFunction[I: ControlInputBatch, S: StateBatch, C: Costs] = CostFunction[
        I, S, C
    ]
    type Error[T: int, M: int] = Error[T, M]
    type ContouringCost[I: ControlInputBatch, S: StateBatch, C: Costs, D: Error] = (
        ContouringCost[I, S, C, D]
    )

    class bicycle:
        type D_x = BicycleD_x
        type D_u = BicycleD_u
        type State = BicycleState
        type StateBatch[T: int = int, M: int = int] = BicycleStateBatch[T, M]
        type Positions[T: int = int, M: int = int] = BicyclePositions[T, M]
        type ControlInputSequence[T: int = int] = BicycleControlInputSequence[T]
        type ControlInputBatch[T: int = int, M: int = int] = BicycleControlInputBatch[
            T, M
        ]

        D_X: Final = BICYCLE_D_X
        D_U: Final = BICYCLE_D_U

    class augmented:
        type State[P: State, V: State, D_x: int = int] = AugmentedState[P, V, D_x]
        type StateBatch[
            P: StateBatch,
            V: StateBatch,
            T: int = int,
            D_x: int = int,
            M: int = int,
        ] = AugmentedStateBatch[P, V, T, D_x, M]
        type ControlInputSequence[
            P: ControlInputSequence,
            V: ControlInputSequence,
            T: int = int,
            D_u: int = int,
        ] = AugmentedControlInputSequence[P, V, T, D_u]
        type ControlInputBatch[
            P: ControlInputBatch,
            V: ControlInputBatch,
            T: int = int,
            D_u: int = int,
            M: int = int,
        ] = AugmentedControlInputBatch[P, V, T, D_u, M]

        state: Final = AugmentedState
        state_batch: Final = AugmentedStateBatch
        control_input_sequence: Final = AugmentedControlInputSequence
        control_input_batch: Final = AugmentedControlInputBatch

    class numpy:
        type State[D_x: int = int] = NumPyState[D_x]
        type StateBatch[T: int = int, D_x: int = int, M: int = int] = NumPyStateBatch[
            T, D_x, M
        ]
        type ControlInputSequence[T: int = int, D_u: int = int] = (
            NumPyControlInputSequence[T, D_u]
        )
        type ControlInputBatch[T: int = int, D_u: int = int, M: int = int] = (
            NumPyControlInputBatch[T, D_u, M]
        )
        type Costs[T: int = int, M: int = int] = NumPyCosts[T, M]
        type PathParameters[T: int = int, M: int = int] = NumPyPathParameters[T, M]
        type ReferencePoints[T: int = int, M: int = int] = NumPyReferencePoints[T, M]
        type Positions[T: int = int, M: int = int] = NumPyPositions[T, M]

        type CostFunction[
            T: int = int,
            D_u: int = int,
            D_x: int = int,
            M: int = int,
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
            type State[D_x: int = int] = NumPySimpleState[D_x]
            type StateBatch[T: int = int, D_x: int = int, M: int = int] = (
                NumPySimpleStateBatch[T, D_x, M]
            )
            type ControlInputSequence[T: int = int, D_u: int = int] = (
                NumPySimpleControlInputSequence[T, D_u]
            )
            type ControlInputBatch[T: int = int, D_u: int = int, M: int = int] = (
                NumPySimpleControlInputBatch[T, D_u, M]
            )
            type Costs[T: int = int, M: int = int] = NumPySimpleCosts[T, M]

            state: Final = NumPySimpleState
            state_batch: Final = NumPySimpleStateBatch
            control_input_sequence: Final = NumPySimpleControlInputSequence
            control_input_batch: Final = NumPySimpleControlInputBatch
            costs: Final = NumPySimpleCosts

        class bicycle:
            type State = NumPyBicycleState
            type StateBatch[T: int = int, M: int = int] = NumPyBicycleStateBatch[T, M]
            type Positions[T: int = int, M: int = int] = NumPyBicyclePositions[T, M]
            type ControlInputSequence[T: int = int] = NumPyBicycleControlInputSequence[
                T
            ]
            type ControlInputBatch[T: int = int, M: int = int] = (
                NumPyBicycleControlInputBatch[T, M]
            )

            state: Final = NumPyBicycleState.create
            state_batch: Final = NumPyBicycleStateBatch
            positions: Final = NumPyBicyclePositions
            control_input_sequence: Final = NumPyBicycleControlInputSequence
            control_input_batch: Final = NumPyBicycleControlInputBatch

        class augmented:
            type State[P: NumPyState, V: NumPyState, D_x: int = int] = (
                NumPyAugmentedState[P, V, D_x]
            )
            type StateBatch[
                P: NumPyStateBatch,
                V: NumPyStateBatch,
                T: int = int,
                D_x: int = int,
                M: int = int,
            ] = NumPyAugmentedStateBatch[P, V, T, D_x, M]
            type ControlInputSequence[
                P: NumPyControlInputSequence,
                V: NumPyControlInputSequence,
                T: int = int,
                D_u: int = int,
            ] = NumPyAugmentedControlInputSequence[P, V, T, D_u]
            type ControlInputBatch[
                P: NumPyControlInputBatch,
                V: NumPyControlInputBatch,
                T: int = int,
                D_u: int = int,
                M: int = int,
            ] = NumPyAugmentedControlInputBatch[P, V, T, D_u, M]

            state: Final = NumPyAugmentedState
            state_batch: Final = NumPyAugmentedStateBatch
            control_input_sequence: Final = NumPyAugmentedControlInputSequence
            control_input_batch: Final = NumPyAugmentedControlInputBatch

    class jax:
        type State[D_x: int = int] = JaxState[D_x]
        type StateBatch[T: int = int, D_x: int = int, M: int = int] = JaxStateBatch[
            T, D_x, M
        ]
        type ControlInputSequence[T: int = int, D_u: int = int] = (
            JaxControlInputSequence[T, D_u]
        )
        type ControlInputBatch[T: int = int, D_u: int = int, M: int = int] = (
            JaxControlInputBatch[T, D_u, M]
        )
        type Costs[T: int = int, M: int = int] = JaxCosts[T, M]
        type PathParameters[T: int = int, M: int = int] = JaxPathParameters[T, M]
        type ReferencePoints[T: int = int, M: int = int] = JaxReferencePoints[T, M]
        type Positions[T: int = int, M: int = int] = JaxPositions[T, M]

        type CostFunction[
            T: int = int,
            D_u: int = int,
            D_x: int = int,
            M: int = int,
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
            type State[D_x: int = int] = JaxSimpleState[D_x]
            type StateBatch[T: int = int, D_x: int = int, M: int = int] = (
                JaxSimpleStateBatch[T, D_x, M]
            )
            type ControlInputSequence[T: int = int, D_u: int = int] = (
                JaxSimpleControlInputSequence[T, D_u]
            )
            type ControlInputBatch[T: int = int, D_u: int = int, M: int = int] = (
                JaxSimpleControlInputBatch[T, D_u, M]
            )
            type Costs[T: int = int, M: int = int] = JaxSimpleCosts[T, M]

            state: Final = JaxSimpleState
            state_batch: Final = JaxSimpleStateBatch
            control_input_sequence: Final = JaxSimpleControlInputSequence
            control_input_batch: Final = JaxSimpleControlInputBatch
            costs: Final = JaxSimpleCosts

        class bicycle:
            type State = JaxBicycleState
            type StateBatch[T: int = int, M: int = int] = JaxBicycleStateBatch[T, M]
            type Positions[T: int = int, M: int = int] = JaxBicyclePositions[T, M]
            type ControlInputSequence[T: int = int] = JaxBicycleControlInputSequence[T]
            type ControlInputBatch[T: int = int, M: int = int] = (
                JaxBicycleControlInputBatch[T, M]
            )

            state: Final = JaxBicycleState
            state_batch: Final = JaxBicycleStateBatch
            positions: Final = JaxBicyclePositions
            control_input_sequence: Final = JaxBicycleControlInputSequence
            control_input_batch: Final = JaxBicycleControlInputBatch
