from typing import Final, Any

from trajax.types import (
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
    BICYCLE_D_X,
    BICYCLE_D_U,
    BicycleD_x,
    BicycleD_u,
    BicycleState,
    BicycleStateBatch,
    BicyclePositions,
    BicycleControlInputSequence,
    BicycleControlInputBatch,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyHeadings,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
    JaxHeadings,
    D_o as D_o_,
    D_O as D_O_,
    NumPyPathParameterExtractor,
    NumPyPathVelocityExtractor,
    NumPyPositionExtractor,
    JaxPathParameterExtractor,
    JaxPathVelocityExtractor,
    JaxPositionExtractor,
    Error as Error,  # NOTE: Aliased to workaround ruff bug.
    ContouringCost as ContouringCost,
    NumPyObstacleStates,
    NumPySampledObstacleStates,
    NumPyObstacleStateProvider,
    JaxObstacleStates,
    JaxSampledObstacleStates,
    JaxObstacleStateProvider,
    AugmentedState,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
)
from trajax.models import (
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
from trajax.costs import (
    NumPyContouringCost,
    JaxContouringCost,
    NumPyDistance,
    JaxDistance,
)
from trajax.obstacles import (
    NumPyObstaclePositionsAndHeadings,
    JaxObstaclePositionsAndHeadings,
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
    type CostFunction[I, S, C] = CostFunction[I, S, C]
    type Error[T: int = Any, M: int = Any] = Error[T, M]
    type ContouringCost[I, S, E] = ContouringCost[I, S, E]

    class obstacle:
        type D_o = D_o_

        D_O: Final = D_O_

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
        type State[P, V] = AugmentedState[P, V]
        type StateBatch[P, V] = AugmentedStateBatch[P, V]
        type ControlInputSequence[P, V] = AugmentedControlInputSequence[P, V]
        type ControlInputBatch[P, V] = AugmentedControlInputBatch[P, V]

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
        type Headings[T: int = Any, M: int = Any] = NumPyHeadings[T, M]
        type ObstacleStates[T: int = Any, K: int = Any] = NumPyObstacleStates[T, K]
        type SampledObstacleStates[T: int = Any, K: int = Any, N: int = Any] = (
            NumPySampledObstacleStates[T, K, N]
        )
        type Distance[T: int = Any, V: int = Any, M: int = Any, N: int = Any] = (
            NumPyDistance[T, V, M, N]
        )

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
        type PathParameterExtractor[S] = NumPyPathParameterExtractor[S]
        type PathVelocityExtractor[I] = NumPyPathVelocityExtractor[I]
        type PositionExtractor[S] = NumPyPositionExtractor[S]
        type ContouringCost[S] = NumPyContouringCost[S]
        type ObstacleStateProvider[O] = NumPyObstacleStateProvider[O]

        path_parameters: Final = NumPyPathParameters
        reference_points: Final = NumPyReferencePoints.create
        positions: Final = NumPyPositions.create
        headings: Final = NumPyHeadings.create
        distance: Final = NumPyDistance
        obstacle_states: Final = NumPyObstaclePositionsAndHeadings

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
        type Headings[T: int = Any, M: int = Any] = JaxHeadings[T, M]
        type ObstacleStates[T: int = Any, K: int = Any] = JaxObstacleStates[T, K]
        type SampledObstacleStates[T: int = Any, K: int = Any, N: int = Any] = (
            JaxSampledObstacleStates[T, K, N]
        )
        type Distance[T: int = Any, V: int = Any, M: int = Any, N: int = Any] = (
            JaxDistance[T, V, M, N]
        )

        type CostFunction[
            T: int = Any,
            D_u: int = Any,
            D_x: int = Any,
            M: int = Any,
        ] = JaxCostFunction[
            JaxControlInputBatch[T, D_u, M], JaxStateBatch[T, D_x, M], JaxCosts[T, M]
        ]
        type PathParameterExtractor[S] = JaxPathParameterExtractor[S]
        type PathVelocityExtractor[I] = JaxPathVelocityExtractor[I]
        type PositionExtractor[S] = JaxPositionExtractor[S]
        type ContouringCost[S] = JaxContouringCost[S]
        type ObstacleStateProvider[O] = JaxObstacleStateProvider[O]

        path_parameters: Final = JaxPathParameters.create
        reference_points: Final = JaxReferencePoints.create
        positions: Final = JaxPositions.create
        headings: Final = JaxHeadings.create
        distance: Final = JaxDistance
        obstacle_states: Final = JaxObstaclePositionsAndHeadings

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
