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
    UNICYCLE_D_X,
    UNICYCLE_D_U,
    UnicycleD_x,
    UnicycleD_u,
    UnicycleState,
    UnicycleStateSequence,
    UnicycleStateBatch,
    UnicyclePositions,
    UnicycleControlInputSequence,
    UnicycleControlInputBatch,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyHeadings,
    NumPyLateralPositions,
    NumPyLongitudinalPositions,
    NumPyNormals,
    NumPyRisk,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
    JaxHeadings,
    JaxLateralPositions,
    JaxLongitudinalPositions,
    JaxNormals,
    JaxRisk,
    PoseD_o as PoseD_o_,
    POSE_D_O as POSE_D_O_,
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
    NumPySampledObstacleStates,
    NumPySampledObstaclePositions,
    NumPySampledObstacleHeadings,
    NumPySampledObstaclePositionExtractor,
    NumPySampledObstacleHeadingExtractor,
    NumPyObstacleStates,
    NumPyObstacleStatesForTimeStep,
    NumPyObstacleStateSequences,
    NumPyObstacleStateProvider,
    NumPyObstaclePositionExtractor,
    JaxSampledObstacleStates,
    JaxSampledObstaclePositions,
    JaxSampledObstacleHeadings,
    JaxSampledObstaclePositionExtractor,
    JaxSampledObstacleHeadingExtractor,
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
    JaxObstacleStateSequences,
    JaxObstacleStateProvider,
    JaxObstaclePositionExtractor,
    AugmentedState,
    AugmentedStateSequence,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    NumPyCovariance,
    NumPyInitialPositionCovariance,
    NumPyInitialVelocityCovariance,
    NumPyCovarianceProvider,
    NumPyPoseCovariance,
    JaxCovariance,
    JaxInitialPositionCovariance,
    JaxInitialVelocityCovariance,
    JaxCovarianceProvider,
    JaxPoseCovariance,
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
    NumPyBicycleObstacleStates,
    NumPyBicycleObstacleStateSequences,
    NumPyBicycleObstacleControlInputSequences,
    NumPyIntegratorObstacleStateSequences,
    JaxBicycleState,
    JaxBicycleStateSequence,
    JaxBicycleStateBatch,
    JaxBicyclePositions,
    JaxBicycleControlInputSequence,
    JaxBicycleControlInputBatch,
    JaxBicycleObstacleStates,
    JaxBicycleObstacleStateSequences,
    JaxBicycleObstacleControlInputSequences,
    JaxIntegratorObstacleStateSequences,
    NumPyUnicycleState,
    NumPyUnicycleStateSequence,
    NumPyUnicycleStateBatch,
    NumPyUnicyclePositions,
    NumPyUnicycleControlInputSequence,
    NumPyUnicycleControlInputBatch,
    NumPyUnicycleObstacleStates,
    NumPyUnicycleObstacleStateSequences,
    NumPyUnicycleObstacleControlInputSequences,
    JaxUnicycleState,
    JaxUnicycleStateSequence,
    JaxUnicycleStateBatch,
    JaxUnicyclePositions,
    JaxUnicycleControlInputSequence,
    JaxUnicycleControlInputBatch,
    JaxUnicycleObstacleStates,
    JaxUnicycleObstacleStateSequences,
    JaxUnicycleObstacleControlInputSequences,
)
from trajax.costs import (
    NumPyContouringCost,
    JaxContouringCost,
    NumPyDistance,
    JaxDistance,
)
from trajax.obstacles import (
    NumPyObstacleIds,
    NumPySampledObstacle2dPoses,
    NumPyObstacle2dPoses,
    NumPyObstacle2dPosesForTimeStep,
    NumPyObstacle2dPositions,
    NumPyObstacle2dPositionsForTimeStep,
    NumPyObstacleStatesRunningHistory,
    JaxObstacleIds,
    JaxSampledObstacle2dPoses,
    JaxObstacle2dPoses,
    JaxObstacle2dPosesForTimeStep,
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
    NumPySimpleSampledObstacleStates,
    NumPySimpleObstacleStatesForTimeStep,
    NumPySimpleObstacleStates,
    JaxSimpleState,
    JaxSimpleStateSequence,
    JaxSimpleStateBatch,
    JaxSimpleControlInputSequence,
    JaxSimpleControlInputBatch,
    JaxSimpleCosts,
    JaxSimpleSampledObstacleStates,
    JaxSimpleObstacleStatesForTimeStep,
    JaxSimpleObstacleStates,
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
    """Namespace of type aliases for states, controls, costs, and related domain types."""

    type State[D_x: int = Any] = State[D_x]
    type StateBatch[T: int = Any, D_x: int = Any, M: int = Any] = StateBatch[T, D_x, M]
    type ControlInputSequence[T: int = Any, D_u: int = Any] = ControlInputSequence[
        T, D_u
    ]
    type ControlInputBatch[T: int = Any, D_u: int = Any, M: int = Any] = (
        ControlInputBatch[T, D_u, M]
    )
    type Costs[T: int = Any, M: int = Any] = Costs[T, M]
    type CostFunction[InputBatchT, StateBatchT, CostsT] = CostFunction[
        InputBatchT, StateBatchT, CostsT
    ]
    type Error[T: int = Any, M: int = Any] = Error[T, M]
    type Risk[T: int = Any, M: int = Any] = Risk[T, M]

    type ContouringCost[InputBatchT, StateBatchT, ErrorT] = ContouringCost[
        InputBatchT, StateBatchT, ErrorT
    ]
    type RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT] = (
        RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT]
    )

    class obstacle:
        type PoseD_o = PoseD_o_

        POSE_D_O: Final = POSE_D_O_

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

    class unicycle:
        type D_x = UnicycleD_x
        type D_u = UnicycleD_u
        type State = UnicycleState
        type StateSequence = UnicycleStateSequence
        type StateBatch[T: int = Any, M: int = Any] = UnicycleStateBatch[T, M]
        type Positions[T: int = Any, M: int = Any] = UnicyclePositions[T, M]
        type ControlInputSequence[T: int = Any] = UnicycleControlInputSequence[T]
        type ControlInputBatch[T: int = Any, M: int = Any] = UnicycleControlInputBatch[
            T, M
        ]

        D_X: Final = UNICYCLE_D_X
        D_U: Final = UNICYCLE_D_U

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
        type SampledObstacleStates[
            T: int = Any,
            D_o: int = Any,
            K: int = Any,
            N: int = Any,
        ] = NumPySampledObstacleStates[T, D_o, K, N]
        type SampledObstaclePositions[T: int = Any, K: int = Any, N: int = Any] = (
            NumPySampledObstaclePositions[T, K, N]
        )
        type SampledObstacleHeadings[T: int = Any, K: int = Any, N: int = Any] = (
            NumPySampledObstacleHeadings[T, K, N]
        )
        type SampledObstaclePositionExtractor[SampledStatesT] = (
            NumPySampledObstaclePositionExtractor[SampledStatesT]
        )
        type SampledObstacleHeadingExtractor[SampledStatesT] = (
            NumPySampledObstacleHeadingExtractor[SampledStatesT]
        )
        type ObstacleStates[
            T: int = Any,
            D_o: int = Any,
            K: int = Any,
            SingleSampleT = Any,
            ObstacleStatesForTimeStepT = Any,
        ] = NumPyObstacleStates[T, D_o, K, SingleSampleT, ObstacleStatesForTimeStepT]
        type ObstacleStatesForTimeStep[
            D_o: int = Any,
            K: int = Any,
            ObstacleStatesT = Any,
        ] = NumPyObstacleStatesForTimeStep[D_o, K, ObstacleStatesT]
        type ObstacleStateSequences[
            T: int = Any,
            D_o: int = Any,
            K: int = Any,
            SingleSampleT = Any,
        ] = NumPyObstacleStateSequences[T, D_o, K, SingleSampleT]
        type ObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ] = NumPyObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ]
        type SampledObstacle2dPoses[T: int = Any, K: int = Any, N: int = Any] = (
            NumPySampledObstacle2dPoses[T, K, N]
        )
        type Obstacle2dPoses[T: int = Any, K: int = Any] = NumPyObstacle2dPoses[T, K]
        type Obstacle2dPosesForTimeStep[K: int = Any] = NumPyObstacle2dPosesForTimeStep[
            K
        ]
        type Obstacle2dPositions[T: int = Any, K: int = Any] = NumPyObstacle2dPositions[
            T, K
        ]
        type Obstacle2dPositionsForTimeStep[K: int = Any] = (
            NumPyObstacle2dPositionsForTimeStep[K]
        )
        type ObstacleStatesRunningHistory[
            StatesT,
            StatesForTimeStepT: NumPyObstacleStatesForTimeStep,
        ] = NumPyObstacleStatesRunningHistory[StatesT, StatesForTimeStepT]
        type Distance[T: int = Any, V: int = Any, M: int = Any, N: int = Any] = (
            NumPyDistance[T, V, M, N]
        )
        type BoundaryDistance[T: int = Any, M: int = Any] = NumPyBoundaryDistance[T, M]
        type Risk[T: int = Any, M: int = Any] = NumPyRisk[T, M]
        type Covariance[T: int = Any, D_o: int = Any, K: int = Any] = NumPyCovariance[
            T, D_o, K
        ]
        type InitialPositionCovariance[K: int = Any] = NumPyInitialPositionCovariance[K]
        type InitialVelocityCovariance[K: int = Any] = NumPyInitialVelocityCovariance[K]
        type PoseCovariance[T: int = Any, K: int = Any] = NumPyPoseCovariance[T, K]

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
        type PathParameterExtractor[StateBatchT] = NumPyPathParameterExtractor[
            StateBatchT
        ]
        type PathVelocityExtractor[InputBatchT] = NumPyPathVelocityExtractor[
            InputBatchT
        ]
        type PositionExtractor[StateBatchT] = NumPyPositionExtractor[StateBatchT]
        type DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT] = (
            NumPyDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT]
        )
        type BoundaryDistanceExtractor[StateBatchT, DistanceT] = (
            NumPyBoundaryDistanceExtractor[StateBatchT, DistanceT]
        )
        type RiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT] = (
            NumPyRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT]
        )
        type ContouringCost[StateBatchT] = NumPyContouringCost[StateBatchT]
        type ObstacleStateProvider[ObstacleStatesT] = NumPyObstacleStateProvider[
            ObstacleStatesT
        ]
        type CovarianceProvider[StateSequencesT, StateCovarianceT, InputCovarianceT] = (
            NumPyCovarianceProvider[StateSequencesT, StateCovarianceT, InputCovarianceT]
        )

        path_parameters: Final = NumPyPathParameters
        reference_points: Final = NumPyReferencePoints.create
        positions: Final = NumPyPositions.create
        headings: Final = NumPyHeadings.create
        lateral_positions: Final = NumPyLateralPositions.create
        longitudinal_positions: Final = NumPyLongitudinalPositions.create
        normals: Final = NumPyNormals.create
        distance: Final = NumPyDistance
        boundary_distance: Final = NumPyBoundaryDistance
        obstacle_ids: Final = NumPyObstacleIds
        obstacle_2d_poses: Final = NumPyObstacle2dPoses
        obstacle_2d_poses_for_time_step: Final = NumPyObstacle2dPosesForTimeStep
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
            type SampledObstacleStates[
                T: int = Any,
                D_o: int = Any,
                K: int = Any,
                N: int = Any,
            ] = NumPySimpleSampledObstacleStates[T, D_o, K, N]
            type ObstacleStatesForTimeStep[D_o: int = Any, K: int = Any] = (
                NumPySimpleObstacleStatesForTimeStep[D_o, K]
            )
            type ObstacleStates[T: int = Any, D_o: int = Any, K: int = Any] = (
                NumPySimpleObstacleStates[T, D_o, K]
            )

            state: Final = NumPySimpleState
            state_sequence: Final = NumPySimpleStateSequence
            state_batch: Final = NumPySimpleStateBatch
            control_input_sequence: Final = NumPySimpleControlInputSequence
            control_input_batch: Final = NumPySimpleControlInputBatch
            costs: Final = NumPySimpleCosts
            obstacle_states: Final = NumPySimpleObstacleStates

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
            type ObstacleStates[K: int = Any] = NumPyBicycleObstacleStates[K]
            type ObstacleStateSequences[T: int = Any, K: int = Any] = (
                NumPyBicycleObstacleStateSequences[T, K]
            )
            type ObstacleControlInputSequences[T: int = Any, K: int = Any] = (
                NumPyBicycleObstacleControlInputSequences[T, K]
            )

            state: Final = NumPyBicycleState
            state_sequence: Final = NumPyBicycleStateSequence
            state_batch: Final = NumPyBicycleStateBatch
            positions: Final = NumPyBicyclePositions
            control_input_sequence: Final = NumPyBicycleControlInputSequence
            control_input_batch: Final = NumPyBicycleControlInputBatch
            obstacle_states: Final = NumPyBicycleObstacleStates
            obstacle_state_sequences: Final = NumPyBicycleObstacleStateSequences
            obstacle_control_input_sequences: Final = (
                NumPyBicycleObstacleControlInputSequences
            )

        class unicycle:
            type State = NumPyUnicycleState
            type StateSequence[T: int = Any] = NumPyUnicycleStateSequence[T]
            type StateBatch[T: int = Any, M: int = Any] = NumPyUnicycleStateBatch[T, M]
            type Positions[T: int = Any, M: int = Any] = NumPyUnicyclePositions[T, M]
            type ControlInputSequence[T: int = Any] = NumPyUnicycleControlInputSequence[
                T
            ]
            type ControlInputBatch[T: int = Any, M: int = Any] = (
                NumPyUnicycleControlInputBatch[T, M]
            )
            type ObstacleStates[K: int = Any] = NumPyUnicycleObstacleStates[K]
            type ObstacleStateSequences[T: int = Any, K: int = Any] = (
                NumPyUnicycleObstacleStateSequences[T, K]
            )
            type ObstacleControlInputSequences[T: int = Any, K: int = Any] = (
                NumPyUnicycleObstacleControlInputSequences[T, K]
            )

            state: Final = NumPyUnicycleState
            state_sequence: Final = NumPyUnicycleStateSequence
            state_batch: Final = NumPyUnicycleStateBatch
            positions: Final = NumPyUnicyclePositions
            control_input_sequence: Final = NumPyUnicycleControlInputSequence
            control_input_batch: Final = NumPyUnicycleControlInputBatch
            obstacle_states: Final = NumPyUnicycleObstacleStates
            obstacle_state_sequences: Final = NumPyUnicycleObstacleStateSequences
            obstacle_control_input_sequences: Final = (
                NumPyUnicycleObstacleControlInputSequences
            )

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
        type SampledObstacleStates[
            T: int = Any,
            D_o: int = Any,
            K: int = Any,
            N: int = Any,
        ] = JaxSampledObstacleStates[T, D_o, K, N]
        type SampledObstaclePositions[T: int = Any, K: int = Any, N: int = Any] = (
            JaxSampledObstaclePositions[T, K, N]
        )
        type SampledObstacleHeadings[T: int = Any, K: int = Any, N: int = Any] = (
            JaxSampledObstacleHeadings[T, K, N]
        )
        type SampledObstaclePositionExtractor[SampledStatesT] = (
            JaxSampledObstaclePositionExtractor[SampledStatesT]
        )
        type SampledObstacleHeadingExtractor[SampledStatesT] = (
            JaxSampledObstacleHeadingExtractor[SampledStatesT]
        )
        type ObstacleStates[
            T: int = Any,
            D_o: int = Any,
            K: int = Any,
            SingleSampleT = Any,
            ObstacleStatesForTimeStepT = Any,
        ] = JaxObstacleStates[T, D_o, K, SingleSampleT, ObstacleStatesForTimeStepT]
        type ObstacleStatesForTimeStep[
            D_o: int = Any,
            K: int = Any,
            ObstacleStatesT = Any,
            NumPyT = Any,
        ] = JaxObstacleStatesForTimeStep[D_o, K, ObstacleStatesT, NumPyT]
        type ObstacleStateSequences[
            T: int = Any,
            D_o: int = Any,
            K: int = Any,
            SingleSampleT = Any,
        ] = JaxObstacleStateSequences[T, D_o, K, SingleSampleT]
        type ObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ] = JaxObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ]
        type SampledObstacle2dPoses[T: int = Any, K: int = Any, N: int = Any] = (
            JaxSampledObstacle2dPoses[T, K, N]
        )
        type Obstacle2dPoses[T: int = Any, K: int = Any] = JaxObstacle2dPoses[T, K]
        type Obstacle2dPosesForTimeStep[K: int = Any] = JaxObstacle2dPosesForTimeStep[K]
        type Obstacle2dPositions[T: int = Any, K: int = Any] = JaxObstacle2dPositions[
            T, K
        ]
        type Obstacle2dPositionsForTimeStep[K: int = Any] = (
            JaxObstacle2dPositionsForTimeStep[K]
        )
        type ObstacleStatesRunningHistory[
            StatesT: JaxObstacleStates,
            StatesForTimeStepT: JaxObstacleStatesForTimeStep,
        ] = JaxObstacleStatesRunningHistory[StatesT, StatesForTimeStepT]
        type Distance[T: int = Any, V: int = Any, M: int = Any, N: int = Any] = (
            JaxDistance[T, V, M, N]
        )
        type BoundaryDistance[T: int = Any, M: int = Any] = JaxBoundaryDistance[T, M]
        type Risk[T: int = Any, M: int = Any] = JaxRisk[T, M]
        type Covariance[T: int = Any, D_o: int = Any, K: int = Any] = JaxCovariance[
            T, D_o, K
        ]
        type InitialPositionCovariance[K: int = Any] = JaxInitialPositionCovariance[K]
        type InitialVelocityCovariance[K: int = Any] = JaxInitialVelocityCovariance[K]
        type PoseCovariance[T: int = Any, K: int = Any] = JaxPoseCovariance[T, K]

        type CostFunction[
            T: int = Any,
            D_u: int = Any,
            D_x: int = Any,
            M: int = Any,
        ] = JaxCostFunction[
            JaxControlInputBatch[T, D_u, M], JaxStateBatch[T, D_x, M], JaxCosts[T, M]
        ]
        type PathParameterExtractor[StateBatchT] = JaxPathParameterExtractor[
            StateBatchT
        ]
        type PathVelocityExtractor[InputBatchT] = JaxPathVelocityExtractor[InputBatchT]
        type PositionExtractor[StateBatchT] = JaxPositionExtractor[StateBatchT]
        type DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT] = (
            JaxDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT]
        )
        type BoundaryDistanceExtractor[StateBatchT, DistanceT] = (
            JaxBoundaryDistanceExtractor[StateBatchT, DistanceT]
        )
        type RiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT] = (
            JaxRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT]
        )
        type ContouringCost[StateBatchT] = JaxContouringCost[StateBatchT]
        type ObstacleStateProvider[ObstacleStatesT] = JaxObstacleStateProvider[
            ObstacleStatesT
        ]
        type CovarianceProvider[StateSequencesT, StateCovarianceT, InputCovarianceT] = (
            JaxCovarianceProvider[StateSequencesT, StateCovarianceT, InputCovarianceT]
        )

        path_parameters: Final = JaxPathParameters.create
        reference_points: Final = JaxReferencePoints.create
        positions: Final = JaxPositions.create
        headings: Final = JaxHeadings.create
        lateral_positions: Final = JaxLateralPositions.create
        longitudinal_positions: Final = JaxLongitudinalPositions.create
        normals: Final = JaxNormals.create
        distance: Final = JaxDistance
        boundary_distance: Final = JaxBoundaryDistance
        obstacle_ids: Final = JaxObstacleIds
        obstacle_2d_poses: Final = JaxObstacle2dPoses
        obstacle_2d_poses_for_time_step: Final = JaxObstacle2dPosesForTimeStep
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
            type SampledObstacleStates[
                T: int = Any,
                D_o: int = Any,
                K: int = Any,
                N: int = Any,
            ] = JaxSimpleSampledObstacleStates[T, D_o, K, N]
            type ObstacleStatesForTimeStep[D_o: int = Any, K: int = Any] = (
                JaxSimpleObstacleStatesForTimeStep[D_o, K]
            )
            type ObstacleStates[T: int = Any, D_o: int = Any, K: int = Any] = (
                JaxSimpleObstacleStates[T, D_o, K]
            )

            state: Final = JaxSimpleState
            state_sequence: Final = JaxSimpleStateSequence
            state_batch: Final = JaxSimpleStateBatch
            control_input_sequence: Final = JaxSimpleControlInputSequence
            control_input_batch: Final = JaxSimpleControlInputBatch
            costs: Final = JaxSimpleCosts
            obstacle_states: Final = JaxSimpleObstacleStates

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
            type ObstacleStates[K: int = Any] = JaxBicycleObstacleStates[K]
            type ObstacleStateSequences[T: int = Any, K: int = Any] = (
                JaxBicycleObstacleStateSequences[T, K]
            )
            type ObstacleControlInputSequences[T: int = Any, K: int = Any] = (
                JaxBicycleObstacleControlInputSequences[T, K]
            )

            state: Final = JaxBicycleState
            state_sequence: Final = JaxBicycleStateSequence
            state_batch: Final = JaxBicycleStateBatch
            positions: Final = JaxBicyclePositions
            control_input_sequence: Final = JaxBicycleControlInputSequence
            control_input_batch: Final = JaxBicycleControlInputBatch
            obstacle_states: Final = JaxBicycleObstacleStates
            obstacle_state_sequences: Final = JaxBicycleObstacleStateSequences
            obstacle_control_input_sequences: Final = (
                JaxBicycleObstacleControlInputSequences
            )

        class unicycle:
            type State = JaxUnicycleState
            type StateSequence[T: int = Any] = JaxUnicycleStateSequence[T]
            type StateBatch[T: int = Any, M: int = Any] = JaxUnicycleStateBatch[T, M]
            type Positions[T: int = Any, M: int = Any] = JaxUnicyclePositions[T, M]
            type ControlInputSequence[T: int = Any] = JaxUnicycleControlInputSequence[T]
            type ControlInputBatch[T: int = Any, M: int = Any] = (
                JaxUnicycleControlInputBatch[T, M]
            )
            type ObstacleStates[K: int = Any] = JaxUnicycleObstacleStates[K]
            type ObstacleStateSequences[T: int = Any, K: int = Any] = (
                JaxUnicycleObstacleStateSequences[T, K]
            )
            type ObstacleControlInputSequences[T: int = Any, K: int = Any] = (
                JaxUnicycleObstacleControlInputSequences[T, K]
            )

            state: Final = JaxUnicycleState
            state_sequence: Final = JaxUnicycleStateSequence
            state_batch: Final = JaxUnicycleStateBatch
            positions: Final = JaxUnicyclePositions
            control_input_sequence: Final = JaxUnicycleControlInputSequence
            control_input_batch: Final = JaxUnicycleControlInputBatch
            obstacle_states: Final = JaxUnicycleObstacleStates
            obstacle_state_sequences: Final = JaxUnicycleObstacleStateSequences
            obstacle_control_input_sequences: Final = (
                JaxUnicycleObstacleControlInputSequences
            )

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
    """Namespace of concrete protocol classes for states, controls, costs, and related types."""

    State: Final = State
    StateSequence: Final = StateSequence
    StateBatch: Final = StateBatch
    ControlInputSequence: Final = ControlInputSequence
    ControlInputBatch: Final = ControlInputBatch
    Costs: Final = Costs
    CostFunction: Final = CostFunction
    Error: Final = Error
    Risk: Final = Risk
    RiskMetric: Final = RiskMetric
    ContouringCost: Final = ContouringCost

    class bicycle:
        State: Final = BicycleState
        StateSequence: Final = BicycleStateSequence
        StateBatch: Final = BicycleStateBatch
        Positions: Final = BicyclePositions
        ControlInputSequence: Final = BicycleControlInputSequence
        ControlInputBatch: Final = BicycleControlInputBatch

    class unicycle:
        State: Final = UnicycleState
        StateSequence: Final = UnicycleStateSequence
        StateBatch: Final = UnicycleStateBatch
        Positions: Final = UnicyclePositions
        ControlInputSequence: Final = UnicycleControlInputSequence
        ControlInputBatch: Final = UnicycleControlInputBatch

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
        LateralPositions: Final = NumPyLateralPositions
        LongitudinalPositions: Final = NumPyLongitudinalPositions
        ObstacleIds: Final = NumPyObstacleIds
        ObstacleStates: Final = NumPyObstacleStates
        ObstacleStatesForTimeStep: Final = NumPyObstacleStatesForTimeStep
        SampledObstacleStates: Final = NumPySampledObstacleStates
        SampledObstaclePositions: Final = NumPySampledObstaclePositions
        SampledObstacleHeadings: Final = NumPySampledObstacleHeadings
        SampledObstaclePositionExtractor: Final = NumPySampledObstaclePositionExtractor
        SampledObstacleHeadingExtractor: Final = NumPySampledObstacleHeadingExtractor
        SampledObstacle2dPoses: Final = NumPySampledObstacle2dPoses
        Obstacle2dPoses: Final = NumPyObstacle2dPoses
        Obstacle2dPosesForTimeStep: Final = NumPyObstacle2dPosesForTimeStep
        Obstacle2dPositions: Final = NumPyObstacle2dPositions
        Obstacle2dPositionsForTimeStep: Final = NumPyObstacle2dPositionsForTimeStep
        ObstacleStatesRunningHistory: Final = NumPyObstacleStatesRunningHistory
        ObstaclePositionExtractor: Final = NumPyObstaclePositionExtractor
        Distance: Final = NumPyDistance
        BoundaryDistance: Final = NumPyBoundaryDistance
        Risk: Final = NumPyRisk
        InitialPositionCovariance: Final = NumPyInitialPositionCovariance
        InitialVelocityCovariance: Final = NumPyInitialVelocityCovariance
        PositionCovariance: Final = NumPyPoseCovariance
        CostFunction: Final = NumPyCostFunction
        PathParameterExtractor: Final = NumPyPathParameterExtractor
        PathVelocityExtractor: Final = NumPyPathVelocityExtractor
        PositionExtractor: Final = NumPyPositionExtractor
        DistanceExtractor: Final = NumPyDistanceExtractor
        BoundaryDistanceExtractor: Final = NumPyBoundaryDistanceExtractor
        RiskMetric: Final = NumPyRiskMetric
        ContouringCost: Final = NumPyContouringCost
        ObstacleStateProvider: Final = NumPyObstacleStateProvider
        CovarianceProvider: Final = NumPyCovarianceProvider

        class simple:
            State: Final = NumPySimpleState
            StateSequence: Final = NumPySimpleStateSequence
            StateBatch: Final = NumPySimpleStateBatch
            ControlInputSequence: Final = NumPySimpleControlInputSequence
            ControlInputBatch: Final = NumPySimpleControlInputBatch
            Costs: Final = NumPySimpleCosts
            SampledObstacleStates: Final = NumPySimpleSampledObstacleStates
            ObstacleStatesForTimeStep: Final = NumPySimpleObstacleStatesForTimeStep
            ObstacleStates: Final = NumPySimpleObstacleStates

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

        class unicycle:
            State: Final = NumPyUnicycleState
            StateSequence: Final = NumPyUnicycleStateSequence
            StateBatch: Final = NumPyUnicycleStateBatch
            Positions: Final = NumPyUnicyclePositions
            ControlInputSequence: Final = NumPyUnicycleControlInputSequence
            ControlInputBatch: Final = NumPyUnicycleControlInputBatch
            ObstacleStateSequences: Final = NumPyUnicycleObstacleStateSequences

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
        LateralPositions: Final = JaxLateralPositions
        LongitudinalPositions: Final = JaxLongitudinalPositions
        ObstacleIds: Final = JaxObstacleIds
        ObstacleStates: Final = JaxObstacleStates
        ObstacleStatesForTimeStep: Final = JaxObstacleStatesForTimeStep
        SampledObstacleStates: Final = JaxSampledObstacleStates
        SampledObstaclePositions: Final = JaxSampledObstaclePositions
        SampledObstacleHeadings: Final = JaxSampledObstacleHeadings
        SampledObstaclePositionExtractor: Final = JaxSampledObstaclePositionExtractor
        SampledObstacleHeadingExtractor: Final = JaxSampledObstacleHeadingExtractor
        SampledObstacle2dPoses: Final = JaxSampledObstacle2dPoses
        Obstacle2dPoses: Final = JaxObstacle2dPoses
        Obstacle2dPosesForTimeStep: Final = JaxObstacle2dPosesForTimeStep
        Obstacle2dPositions: Final = JaxObstacle2dPositions
        Obstacle2dPositionsForTimeStep: Final = JaxObstacle2dPositionsForTimeStep
        ObstacleStatesRunningHistory: Final = JaxObstacleStatesRunningHistory
        ObstaclePositionExtractor: Final = JaxObstaclePositionExtractor
        Distance: Final = JaxDistance
        BoundaryDistance: Final = JaxBoundaryDistance
        Risk: Final = JaxRisk
        InitialPositionCovariance: Final = JaxInitialPositionCovariance
        InitialVelocityCovariance: Final = JaxInitialVelocityCovariance
        PositionCovariance: Final = JaxPoseCovariance
        CostFunction: Final = JaxCostFunction
        PathParameterExtractor: Final = JaxPathParameterExtractor
        PathVelocityExtractor: Final = JaxPathVelocityExtractor
        PositionExtractor: Final = JaxPositionExtractor
        DistanceExtractor: Final = JaxDistanceExtractor
        BoundaryDistanceExtractor: Final = JaxBoundaryDistanceExtractor
        RiskMetric: Final = JaxRiskMetric
        ContouringCost: Final = JaxContouringCost
        ObstacleStateProvider: Final = JaxObstacleStateProvider
        CovarianceProvider: Final = JaxCovarianceProvider

        class simple:
            State: Final = JaxSimpleState
            StateSequence: Final = JaxSimpleStateSequence
            StateBatch: Final = JaxSimpleStateBatch
            ControlInputSequence: Final = JaxSimpleControlInputSequence
            ControlInputBatch: Final = JaxSimpleControlInputBatch
            Costs: Final = JaxSimpleCosts
            SampledObstacleStates: Final = JaxSimpleSampledObstacleStates
            ObstacleStatesForTimeStep: Final = JaxSimpleObstacleStatesForTimeStep
            ObstacleStates: Final = JaxSimpleObstacleStates

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

        class unicycle:
            State: Final = JaxUnicycleState
            StateSequence: Final = JaxUnicycleStateSequence
            StateBatch: Final = JaxUnicycleStateBatch
            Positions: Final = JaxUnicyclePositions
            ControlInputSequence: Final = JaxUnicycleControlInputSequence
            ControlInputBatch: Final = JaxUnicycleControlInputBatch
            ObstacleStateSequences: Final = JaxUnicycleObstacleStateSequences

        class augmented:
            State: Final = JaxAugmentedState
            StateSequence: Final = JaxAugmentedStateSequence
            StateBatch: Final = JaxAugmentedStateBatch
            ControlInputSequence: Final = JaxAugmentedControlInputSequence
            ControlInputBatch: Final = JaxAugmentedControlInputBatch
