from typing import Final

from trajax.model.common import (
    State as AnyState,
    StateBatch as AnyStateBatch,
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
    State as NumPyState,
    StateBatch as NumPyStateBatch,
    ControlInputBatch as NumPyControlInputBatch,
)
from trajax.model.bicycle.common import (
    D_X as BICYCLE_D_X,
    D_U as BICYCLE_D_U,
    D_x as BicycleD_x,
    D_u as BicycleD_u,
    State as BicycleState,
    StateBatch as BicycleStateBatch,
    Positions as BicyclePositions,
    ControlInputSequence as BicycleControlInputSequence,
    ControlInputBatch as BicycleControlInputBatch,
)
from trajax.model.bicycle.basic import (
    State as NumPyBicycleState,
    StateBatch as NumPyBicycleStateBatch,
    Positions as NumPyBicyclePositions,
    ControlInputSequence as NumPyBicycleControlInputSequence,
    ControlInputBatch as NumPyBicycleControlInputBatch,
)
from trajax.model.bicycle.accelerated import (
    State as JaxBicycleState,
    StateBatch as JaxBicycleStateBatch,
    Positions as JaxBicyclePositions,
    ControlInputSequence as JaxBicycleControlInputSequence,
    ControlInputBatch as JaxBicycleControlInputBatch,
)
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
from trajax.mppi.common import Costs as AnyCosts
from trajax.mppi.basic import (
    ControlInputSequence as NumPyControlInputSequence,
    Costs as NumPyCosts,
    CostFunction as NumPyCostFunction,
)
from trajax.mppi.accelerated import (
    State as JaxState,
    StateBatch as JaxStateBatch,
    ControlInputSequence as JaxControlInputSequence,
    ControlInputBatch as JaxControlInputBatch,
    Costs as JaxCosts,
    CostFunction as JaxCostFunction,
)
from trajax.costs.basic import (
    PathParameterExtractor as NumPyPathParameterExtractor,
    PathVelocityExtractor as NumPyPathVelocityExtractor,
    PositionExtractor as NumPyPositionExtractor,
)
from trajax.costs.accelerated import (
    PathParameterExtractor as JaxPathParameterExtractor,
    PathVelocityExtractor as JaxPathVelocityExtractor,
    PositionExtractor as JaxPositionExtractor,
)
from trajax.states.simple.basic import (
    State as NumPySimpleState,
    StateBatch as NumPySimpleStateBatch,
    ControlInputSequence as NumPySimpleControlInputSequence,
    ControlInputBatch as NumPySimpleControlInputBatch,
    Costs as NumPySimpleCosts,
)
from trajax.states.simple.accelerated import (
    State as JaxSimpleState,
    StateBatch as JaxSimpleStateBatch,
    ControlInputSequence as JaxSimpleControlInputSequence,
    ControlInputBatch as JaxSimpleControlInputBatch,
    Costs as JaxSimpleCosts,
)
from trajax.states.augmented.common import (
    AugmentedState as AugmentedState,
    AugmentedStateBatch as AugmentedStateBatch,
    AugmentedControlInputSequence as AugmentedControlInputSequence,
    AugmentedControlInputBatch as AugmentedControlInputBatch,
    AugmentedState as NumPyAugmentedState,
    AugmentedStateBatch as NumPyAugmentedStateBatch,
    AugmentedControlInputBatch as NumPyAugmentedControlInputBatch,
)
from trajax.states.augmented.basic import (
    AugmentedControlInputSequence as NumPyAugmentedControlInputSequence,
)


class types:
    type State[D_x: int = int] = AnyState[D_x]
    type StateBatch[T: int = int, D_x: int = int, M: int = int] = AnyStateBatch[
        T, D_x, M
    ]
    type ControlInputSequence[T: int = int, D_u: int = int] = AnyControlInputSequence[
        T, D_u
    ]
    type ControlInputBatch[T: int = int, D_u: int = int, M: int = int] = (
        AnyControlInputBatch[T, D_u, M]
    )
    type Costs[T: int = int, M: int = int] = AnyCosts[T, M]

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
        type State[P: AnyState, V: AnyState, D_x: int = int] = AugmentedState[P, V, D_x]
        type StateBatch[
            P: AnyStateBatch,
            V: AnyStateBatch,
            T: int = int,
            D_x: int = int,
            M: int = int,
        ] = AugmentedStateBatch[P, V, T, D_x, M]
        type ControlInputSequence[
            P: AnyControlInputSequence,
            V: AnyControlInputSequence,
            T: int = int,
            D_u: int = int,
        ] = AugmentedControlInputSequence[P, V, T, D_u]
        type ControlInputBatch[
            P: AnyControlInputBatch,
            V: AnyControlInputBatch,
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
        type PathParameterExtractor[StateT: NumPyStateBatch] = (
            NumPyPathParameterExtractor[StateT]
        )
        type PathVelocityExtractor[InputT: NumPyControlInputBatch] = (
            NumPyPathVelocityExtractor[InputT]
        )
        type PositionExtractor[StateT: NumPyStateBatch] = NumPyPositionExtractor[StateT]

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
        type PathParameterExtractor[StateT: JaxStateBatch] = JaxPathParameterExtractor[
            StateT
        ]
        type PathVelocityExtractor[InputT: JaxControlInputBatch] = (
            JaxPathVelocityExtractor[InputT]
        )
        type PositionExtractor[StateT: JaxStateBatch] = JaxPositionExtractor[StateT]

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
