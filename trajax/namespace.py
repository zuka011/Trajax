from typing import Final, Any

from trajax.types import (
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPyCostFunction,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxCosts,
    JaxCostFunction,
    State as State,
    StateSequence as StateSequence,
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
    BicycleStateSequence,
    BicycleStateBatch,
    BicyclePositions,
    BicycleControlInputSequence,
    BicycleControlInputBatch,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyHeadings,
    NumPyLateralPositions,
    NumPyLongitudinalPositions,
    NumPyRisk,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
    JaxHeadings,
    JaxLateralPositions,
    JaxLongitudinalPositions,
    JaxRisk,
    D_o as D_o_,
    D_O as D_O_,
    NumPyPathParameterExtractor,
    NumPyPathVelocityExtractor,
    NumPyPositionExtractor,
    NumPyDistanceExtractor,
    NumPyRiskMetric,
    JaxPathParameterExtractor,
    JaxPathVelocityExtractor,
    JaxPositionExtractor,
    JaxDistanceExtractor,
    JaxRiskMetric,
    Error as Error,  # NOTE: Aliased to workaround ruff bug.
    Risk as Risk,
    RiskMetric as RiskMetric,
    ContouringCost as ContouringCost,
    NumPyObstacleStateProvider,
    NumPyObstaclePositionExtractor,
    JaxObstacleStateProvider,
    JaxObstaclePositionExtractor,
    AugmentedState,
    AugmentedStateSequence,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    NumPyInitialPositionCovariance,
    NumPyInitialVelocityCovariance,
    NumPyInitialCovarianceProvider,
    NumPyPositionCovariance,
    JaxInitialPositionCovariance,
    JaxInitialVelocityCovariance,
    JaxInitialCovarianceProvider,
    JaxPositionCovariance,
    NumPyBoundaryDistance,
    NumPyBoundaryDistanceExtractor,
    JaxBoundaryDistance,
    JaxBoundaryDistanceExtractor,
)
from trajax.models import (
    NumPyBicycleState,
    NumPyBicycleStateSequence,
    NumPyBicycleStateBatch,
    NumPyBicyclePositions,
    NumPyBicycleControlInputSequence,
    NumPyBicycleControlInputBatch,
    NumPyBicycleObstacleStateSequences,
    NumPyIntegratorObstacleStateSequences,
    JaxBicycleState,
    JaxBicycleStateSequence,
    JaxBicycleStateBatch,
    JaxBicyclePositions,
    JaxBicycleControlInputSequence,
    JaxBicycleControlInputBatch,
    JaxBicycleObstacleStateSequences,
    JaxIntegratorObstacleStateSequences,
)
from trajax.costs import (
    NumPyContouringCost,
    JaxContouringCost,
    NumPyDistance,
    JaxDistance,
)
from trajax.obstacles import (
    NumPySampledObstacleStates,
    NumPyObstacleIds,
    NumPyObstacleStates,
    NumPyObstacleStatesForTimeStep,
    NumPyObstacle2dPositions,
    NumPyObstacle2dPositionsForTimeStep,
    NumPyObstacleStatesRunningHistory,
    JaxSampledObstacleStates,
    JaxObstacleIds,
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
    JaxObstacle2dPositions,
    JaxObstacle2dPositionsForTimeStep,
    JaxObstacleStatesRunningHistory,
)
from trajax.states import (
    NumPySimpleState,
    NumPySimpleStateSequence,
    NumPySimpleStateBatch,
    NumPySimpleControlInputSequence,
    NumPySimpleControlInputBatch,
    NumPySimpleCosts,
    JaxSimpleState,
    JaxSimpleStateSequence,
    JaxSimpleStateBatch,
    JaxSimpleControlInputSequence,
    JaxSimpleControlInputBatch,
    JaxSimpleCosts,
    NumPyAugmentedState,
    NumPyAugmentedStateSequence,
    NumPyAugmentedStateBatch,
    NumPyAugmentedControlInputSequence,
    NumPyAugmentedControlInputBatch,
    JaxAugmentedState,
    JaxAugmentedStateSequence,
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
    type Risk[T: int = Any, M: int = Any] = Risk[T, M]

    type ContouringCost[I, S, E] = ContouringCost[I, S, E]
    type RiskMetric[CF, SB, OS, S, R] = RiskMetric[CF, SB, OS, S, R]

    class obstacle:
        type D_o = D_o_

        D_O: Final = D_O_

    class bicycle:
        type D_x = BicycleD_x
        type D_u = BicycleD_u
        type State = BicycleState
        type StateSequence = BicycleStateSequence
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
        type StateSequence[P, V] = AugmentedStateSequence[P, V]
        type StateBatch[P, V] = AugmentedStateBatch[P, V]
        type ControlInputSequence[P, V] = AugmentedControlInputSequence[P, V]
        type ControlInputBatch[P, V] = AugmentedControlInputBatch[P, V]

        state: Final = AugmentedState
        state_sequence: Final = AugmentedStateSequence
        state_batch: Final = AugmentedStateBatch
        control_input_sequence: Final = AugmentedControlInputSequence
        control_input_batch: Final = AugmentedControlInputBatch

    class numpy:
        type State[D_x: int = Any] = NumPyState[D_x]
        type StateSequence[T: int = Any, D_x: int = Any] = NumPyStateSequence[T, D_x]
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
        type LateralPositions[T: int = Any, M: int = Any] = NumPyLateralPositions[T, M]
        type LongitudinalPositions[T: int = Any, M: int = Any] = (
            NumPyLongitudinalPositions[T, M]
        )
        type ObstacleIds[K: int = Any] = NumPyObstacleIds[K]
        type ObstacleStates[T: int = Any, K: int = Any] = NumPyObstacleStates[T, K]
        type SampledObstacleStates[T: int = Any, K: int = Any, N: int = Any] = (
            NumPySampledObstacleStates[T, K, N]
        )
        type ObstacleStatesForTimeStep[K: int = Any] = NumPyObstacleStatesForTimeStep[K]
        type Obstacle2dPositions[T: int = Any, K: int = Any] = NumPyObstacle2dPositions[
            T, K
        ]
        type Obstacle2dPositionsForTimeStep[K: int = Any] = (
            NumPyObstacle2dPositionsForTimeStep[K]
        )
        type ObstacleStatesRunningHistory[T: int = Any, K: int = Any] = (
            NumPyObstacleStatesRunningHistory[T, K]
        )
        type Distance[T: int = Any, V: int = Any, M: int = Any, N: int = Any] = (
            NumPyDistance[T, V, M, N]
        )
        type BoundaryDistance[T: int = Any, M: int = Any] = NumPyBoundaryDistance[T, M]
        type Risk[T: int = Any, M: int = Any] = NumPyRisk[T, M]
        type InitialPositionCovariance[K: int = Any] = NumPyInitialPositionCovariance[K]
        type InitialVelocityCovariance[K: int = Any] = NumPyInitialVelocityCovariance[K]
        type PositionCovariance[T: int = Any, K: int = Any] = NumPyPositionCovariance[
            T, K
        ]

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
        type DistanceExtractor[SB, SOS, D] = NumPyDistanceExtractor[SB, SOS, D]
        type BoundaryDistanceExtractor[SB, D] = NumPyBoundaryDistanceExtractor[SB, D]
        type RiskMetric[SB, OS, SOS] = NumPyRiskMetric[SB, OS, SOS]
        type ContouringCost[S] = NumPyContouringCost[S]
        type ObstacleStateProvider[O] = NumPyObstacleStateProvider[O]
        type ObstaclePositionExtractor[OTS, O, PTS, P] = NumPyObstaclePositionExtractor[
            OTS, O, PTS, P
        ]
        type InitialCovarianceProvider[S] = NumPyInitialCovarianceProvider[S]

        path_parameters: Final = NumPyPathParameters
        reference_points: Final = NumPyReferencePoints.create
        positions: Final = NumPyPositions.create
        headings: Final = NumPyHeadings.create
        lateral_positions: Final = NumPyLateralPositions.create
        longitudinal_positions: Final = NumPyLongitudinalPositions.create
        distance: Final = NumPyDistance
        boundary_distance: Final = NumPyBoundaryDistance
        obstacle_ids: Final = NumPyObstacleIds
        obstacle_states: Final = NumPyObstacleStates
        obstacle_states_for_time_step: Final = NumPyObstacleStatesForTimeStep
        obstacle_states_running_history: Final = NumPyObstacleStatesRunningHistory

        class simple:
            type State[D_x: int = Any] = NumPySimpleState[D_x]
            type StateSequence[T: int = Any, D_x: int = Any] = NumPySimpleStateSequence[
                T, D_x
            ]
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
            state_sequence: Final = NumPySimpleStateSequence
            state_batch: Final = NumPySimpleStateBatch
            control_input_sequence: Final = NumPySimpleControlInputSequence
            control_input_batch: Final = NumPySimpleControlInputBatch
            costs: Final = NumPySimpleCosts

        class integrator:
            type ObstacleStateSequences[T: int = Any, D_o: int = Any, K: int = Any] = (
                NumPyIntegratorObstacleStateSequences[T, D_o, K]
            )

        class bicycle:
            type State = NumPyBicycleState
            type StateSequence[T: int = Any] = NumPyBicycleStateSequence[T]
            type StateBatch[T: int = Any, M: int = Any] = NumPyBicycleStateBatch[T, M]
            type Positions[T: int = Any, M: int = Any] = NumPyBicyclePositions[T, M]
            type ControlInputSequence[T: int = Any] = NumPyBicycleControlInputSequence[
                T
            ]
            type ControlInputBatch[T: int = Any, M: int = Any] = (
                NumPyBicycleControlInputBatch[T, M]
            )
            type ObstacleStateSequences[T: int = Any, K: int = Any] = (
                NumPyBicycleObstacleStateSequences[T, K]
            )

            state: Final = NumPyBicycleState
            state_sequence: Final = NumPyBicycleStateSequence
            state_batch: Final = NumPyBicycleStateBatch
            positions: Final = NumPyBicyclePositions
            control_input_sequence: Final = NumPyBicycleControlInputSequence
            control_input_batch: Final = NumPyBicycleControlInputBatch

        class augmented:
            type State[P: NumPyState, V: NumPyState] = NumPyAugmentedState[P, V]
            type StateSequence[P: NumPyStateSequence, V: NumPyStateSequence] = (
                NumPyAugmentedStateSequence[P, V]
            )
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
            state_sequence: Final = NumPyAugmentedStateSequence
            state_batch: Final = NumPyAugmentedStateBatch
            control_input_sequence: Final = NumPyAugmentedControlInputSequence
            control_input_batch: Final = NumPyAugmentedControlInputBatch

    class jax:
        type State[D_x: int = Any] = JaxState[D_x]
        type StateSequence[T: int = Any, D_x: int = Any] = JaxStateSequence[T, D_x]
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
        type LateralPositions[T: int = Any, M: int = Any] = JaxLateralPositions[T, M]
        type LongitudinalPositions[T: int = Any, M: int = Any] = (
            JaxLongitudinalPositions[T, M]
        )
        type ObstacleIds[K: int = Any] = JaxObstacleIds[K]
        type ObstacleStates[T: int = Any, K: int = Any] = JaxObstacleStates[T, K]
        type SampledObstacleStates[T: int = Any, K: int = Any, N: int = Any] = (
            JaxSampledObstacleStates[T, K, N]
        )
        type ObstacleStatesForTimeStep[K: int = Any] = JaxObstacleStatesForTimeStep[K]
        type Obstacle2dPositions[T: int = Any, K: int = Any] = JaxObstacle2dPositions[
            T, K
        ]
        type Obstacle2dPositionsForTimeStep[K: int = Any] = (
            JaxObstacle2dPositionsForTimeStep[K]
        )
        type ObstacleStatesRunningHistory[T: int = Any, K: int = Any] = (
            JaxObstacleStatesRunningHistory[T, K]
        )
        type Distance[T: int = Any, V: int = Any, M: int = Any, N: int = Any] = (
            JaxDistance[T, V, M, N]
        )
        type BoundaryDistance[T: int = Any, M: int = Any] = JaxBoundaryDistance[T, M]
        type Risk[T: int = Any, M: int = Any] = JaxRisk[T, M]
        type InitialPositionCovariance[K: int = Any] = JaxInitialPositionCovariance[K]
        type InitialVelocityCovariance[K: int = Any] = JaxInitialVelocityCovariance[K]
        type PositionCovariance[T: int = Any, K: int = Any] = JaxPositionCovariance[
            T, K
        ]

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
        type DistanceExtractor[SB, SOS, D] = JaxDistanceExtractor[SB, SOS, D]
        type BoundaryDistanceExtractor[SB, D] = JaxBoundaryDistanceExtractor[SB, D]
        type RiskMetric[SB, OS, SOS] = JaxRiskMetric[SB, OS, SOS]
        type ContouringCost[S] = JaxContouringCost[S]
        type ObstacleStateProvider[O] = JaxObstacleStateProvider[O]
        type ObstaclePositionExtractor[OTS, O, PTS, P] = JaxObstaclePositionExtractor[
            OTS, O, PTS, P
        ]
        type InitialCovarianceProvider[S] = JaxInitialCovarianceProvider[S]

        path_parameters: Final = JaxPathParameters.create
        reference_points: Final = JaxReferencePoints.create
        positions: Final = JaxPositions.create
        headings: Final = JaxHeadings.create
        lateral_positions: Final = JaxLateralPositions.create
        longitudinal_positions: Final = JaxLongitudinalPositions.create
        distance: Final = JaxDistance
        boundary_distance: Final = JaxBoundaryDistance
        obstacle_ids: Final = JaxObstacleIds
        obstacle_states: Final = JaxObstacleStates
        obstacle_states_for_time_step: Final = JaxObstacleStatesForTimeStep
        obstacle_states_running_history: Final = JaxObstacleStatesRunningHistory

        class simple:
            type State[D_x: int = Any] = JaxSimpleState[D_x]
            type StateSequence[T: int = Any, D_x: int = Any] = JaxSimpleStateSequence[
                T, D_x
            ]
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
            state_sequence: Final = JaxSimpleStateSequence
            state_batch: Final = JaxSimpleStateBatch
            control_input_sequence: Final = JaxSimpleControlInputSequence
            control_input_batch: Final = JaxSimpleControlInputBatch
            costs: Final = JaxSimpleCosts

        class integrator:
            type ObstacleStateSequences[T: int = Any, D_o: int = Any, K: int = Any] = (
                JaxIntegratorObstacleStateSequences[T, D_o, K]
            )

        class bicycle:
            type State = JaxBicycleState
            type StateSequence[T: int = Any] = JaxBicycleStateSequence[T]
            type StateBatch[T: int = Any, M: int = Any] = JaxBicycleStateBatch[T, M]
            type Positions[T: int = Any, M: int = Any] = JaxBicyclePositions[T, M]
            type ControlInputSequence[T: int = Any] = JaxBicycleControlInputSequence[T]
            type ControlInputBatch[T: int = Any, M: int = Any] = (
                JaxBicycleControlInputBatch[T, M]
            )
            type ObstacleStateSequences[T: int = Any, K: int = Any] = (
                JaxBicycleObstacleStateSequences[T, K]
            )

            state: Final = JaxBicycleState
            state_sequence: Final = JaxBicycleStateSequence
            state_batch: Final = JaxBicycleStateBatch
            positions: Final = JaxBicyclePositions
            control_input_sequence: Final = JaxBicycleControlInputSequence
            control_input_batch: Final = JaxBicycleControlInputBatch

        class augmented:
            type State[P: JaxState, V: JaxState] = JaxAugmentedState[P, V]
            type StateSequence[P: JaxStateSequence, V: JaxStateSequence] = (
                JaxAugmentedStateSequence[P, V]
            )
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
            state_sequence: Final = JaxAugmentedStateSequence
            state_batch: Final = JaxAugmentedStateBatch
            control_input_sequence: Final = JaxAugmentedControlInputSequence
            control_input_batch: Final = JaxAugmentedControlInputBatch


class classes:
    State: Final = State
    StateSequence: Final = StateSequence
    StateBatch: Final = StateBatch
    ControlInputSequence: Final = ControlInputSequence
    ControlInputBatch: Final = ControlInputBatch
    Costs: Final = Costs
    CostFunction: Final = CostFunction
    Error: Final = Error
    ContouringCost: Final = ContouringCost

    class bicycle:
        State: Final = BicycleState
        StateSequence: Final = BicycleStateSequence
        StateBatch: Final = BicycleStateBatch
        Positions: Final = BicyclePositions
        ControlInputSequence: Final = BicycleControlInputSequence
        ControlInputBatch: Final = BicycleControlInputBatch

    class augmented:
        State: Final = AugmentedState
        StateSequence: Final = AugmentedStateSequence
        StateBatch: Final = AugmentedStateBatch
        ControlInputSequence: Final = AugmentedControlInputSequence
        ControlInputBatch: Final = AugmentedControlInputBatch

    class numpy:
        State: Final = NumPyState
        StateSequence: Final = NumPyStateSequence
        StateBatch: Final = NumPyStateBatch
        ControlInputSequence: Final = NumPyControlInputSequence
        ControlInputBatch: Final = NumPyControlInputBatch
        Costs: Final = NumPyCosts
        PathParameters: Final = NumPyPathParameters
        ReferencePoints: Final = NumPyReferencePoints
        Positions: Final = NumPyPositions
        Headings: Final = NumPyHeadings
        ObstacleStates: Final = NumPyObstacleStates
        SampledObstacleStates: Final = NumPySampledObstacleStates
        Distance: Final = NumPyDistance
        CostFunction: Final = NumPyCostFunction
        PathParameterExtractor: Final = NumPyPathParameterExtractor
        PathVelocityExtractor: Final = NumPyPathVelocityExtractor
        PositionExtractor: Final = NumPyPositionExtractor
        ContouringCost: Final = NumPyContouringCost
        ObstacleStateProvider: Final = NumPyObstacleStateProvider

        class simple:
            State: Final = NumPySimpleState
            StateSequence: Final = NumPySimpleStateSequence
            StateBatch: Final = NumPySimpleStateBatch
            ControlInputSequence: Final = NumPySimpleControlInputSequence
            ControlInputBatch: Final = NumPySimpleControlInputBatch
            Costs: Final = NumPySimpleCosts

        class integrator:
            ObstacleStateSequences: Final = NumPyIntegratorObstacleStateSequences

        class bicycle:
            State: Final = NumPyBicycleState
            StateSequence: Final = NumPyBicycleStateSequence
            StateBatch: Final = NumPyBicycleStateBatch
            Positions: Final = NumPyBicyclePositions
            ControlInputSequence: Final = NumPyBicycleControlInputSequence
            ControlInputBatch: Final = NumPyBicycleControlInputBatch
            ObstacleStateSequences: Final = NumPyBicycleObstacleStateSequences

        class augmented:
            State: Final = NumPyAugmentedState
            StateSequence: Final = NumPyAugmentedStateSequence
            StateBatch: Final = NumPyAugmentedStateBatch
            ControlInputSequence: Final = NumPyAugmentedControlInputSequence
            ControlInputBatch: Final = NumPyAugmentedControlInputBatch

    class jax:
        State: Final = JaxState
        StateSequence: Final = JaxStateSequence
        StateBatch: Final = JaxStateBatch
        ControlInputSequence: Final = JaxControlInputSequence
        ControlInputBatch: Final = JaxControlInputBatch
        Costs: Final = JaxCosts
        PathParameters: Final = JaxPathParameters
        ReferencePoints: Final = JaxReferencePoints
        Positions: Final = JaxPositions
        Headings: Final = JaxHeadings
        ObstacleStates: Final = JaxObstacleStates
        SampledObstacleStates: Final = JaxSampledObstacleStates
        Distance: Final = JaxDistance
        CostFunction: Final = JaxCostFunction
        PathParameterExtractor: Final = JaxPathParameterExtractor
        PathVelocityExtractor: Final = JaxPathVelocityExtractor
        PositionExtractor: Final = JaxPositionExtractor
        ContouringCost: Final = JaxContouringCost
        ObstacleStateProvider: Final = JaxObstacleStateProvider

        class simple:
            State: Final = JaxSimpleState
            StateSequence: Final = JaxSimpleStateSequence
            StateBatch: Final = JaxSimpleStateBatch
            ControlInputSequence: Final = JaxSimpleControlInputSequence
            ControlInputBatch: Final = JaxSimpleControlInputBatch
            Costs: Final = JaxSimpleCosts

        class integrator:
            ObstacleStateSequences: Final = JaxIntegratorObstacleStateSequences

        class bicycle:
            State: Final = JaxBicycleState
            StateSequence: Final = JaxBicycleStateSequence
            StateBatch: Final = JaxBicycleStateBatch
            Positions: Final = JaxBicyclePositions
            ControlInputSequence: Final = JaxBicycleControlInputSequence
            ControlInputBatch: Final = JaxBicycleControlInputBatch
            ObstacleStateSequences: Final = JaxBicycleObstacleStateSequences

        class augmented:
            State: Final = JaxAugmentedState
            StateSequence: Final = JaxAugmentedStateSequence
            StateBatch: Final = JaxAugmentedStateBatch
            ControlInputSequence: Final = JaxAugmentedControlInputSequence
            ControlInputBatch: Final = JaxAugmentedControlInputBatch
