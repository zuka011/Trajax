from trajax import ObstacleIdAssignment, ObstacleIds, types, obstacles

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


type NumPyObstacleStatesForTimeStep = types.numpy.ObstacleStatesForTimeStep
type NumPyObstacleStates = types.numpy.ObstacleStates
type NumPyObstacle2dPositionsForTimeStep = types.numpy.Obstacle2dPositionsForTimeStep
type NumPyObstacle2dPositions = types.numpy.Obstacle2dPositions
type JaxObstacleStatesForTimeStep = types.jax.ObstacleStatesForTimeStep
type JaxObstacleStates = types.jax.ObstacleStates
type JaxObstacle2dPositionsForTimeStep = types.jax.Obstacle2dPositionsForTimeStep
type JaxObstacle2dPositions = types.jax.Obstacle2dPositions


class NumPyObstaclePositionExtractor:
    def of_states_for_time_step(
        self, states: NumPyObstacleStatesForTimeStep, /
    ) -> NumPyObstacle2dPositionsForTimeStep:
        return states.positions()

    def of_states(self, states: NumPyObstacleStates, /) -> NumPyObstacle2dPositions:
        return states.positions()


class JaxObstaclePositionExtractor:
    def of_states_for_time_step(
        self, states: JaxObstacleStatesForTimeStep, /
    ) -> JaxObstacle2dPositionsForTimeStep:
        return states.positions()

    def of_states(self, states: JaxObstacleStates, /) -> JaxObstacle2dPositions:
        return states.positions()


class test_that_ids_are_assigned_to_obstacles:
    def cases(id_assignment, position_extractor, data) -> None:
        return [
            (  # All new obstacles (no history)
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([0.2, 1.5, 3.0], shape=(K := 3,)),
                    y=array([1.2, 2.5, 4.0], shape=(K,)),
                    heading=array([1.0, 2.0, 1.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=np.empty((0, 0)),
                    y=np.empty((0, 0)),
                    heading=np.empty((0, 0)),
                ),
                ids := data.obstacle_ids([]),
                expected := data.obstacle_ids([1, 2, 3]),
            ),
            (  # Single obstacle is tracked
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([0.3], shape=(K := 1,)),
                    y=array([1.3], shape=(K,)),
                    heading=array([1.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=array([[0.2]], shape=(1, 1)),
                    y=array([[1.2]], shape=(1, 1)),
                    heading=array([[1.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([5]),
            ),
            (  # Single obstacle outside cutoff should get a new ID
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([5.0], shape=(K := 1,)),
                    y=array([5.0], shape=(K,)),
                    heading=array([1.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=array([[0.2]], shape=(1, 1)),
                    y=array([[1.2]], shape=(1, 1)),
                    heading=array([[1.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([1]),
            ),
            (  # Multiple well-separated obstacles persist
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([0.3, 10.1], shape=(K := 2,)),
                    y=array([0.1, 10.2], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([3, 7]),
            ),
            (  # New obstacle appears (one matches, one new)
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=10,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([0.3, 50.0], shape=(K := 2,)),
                    y=array([0.1, 50.0], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=array([[0.2]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[0.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([3]),
                expected := data.obstacle_ids([3, 10]),
            ),
            (  # Obstacle disappears
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([10.1], shape=(K := 1,)),
                    y=array([10.2], shape=(K,)),
                    heading=array([0.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([7]),
            ),
            (  # Input order: swapped observation order, IDs follow obstacles
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([10.1, 0.3], shape=(K := 2,)),  # swapped vs history
                    y=array([10.2, 0.1], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([7, 3]),
            ),
            (  # Empty observations
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([], shape=(K := 0,)),
                    y=array([], shape=(K,)),
                    heading=array([], shape=(K,)),
                ),
                history := data.obstacle_states(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([]),
            ),
            (  # History padded with NaN columns (K_history > K_ids)
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_states_for_time_step(
                    x=array([0.3, 10.1], shape=(K := 2,)),
                    y=array([0.1, 10.2], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_states(
                    # 4 columns, but only 2 are valid (rest are NaN padding)
                    x=array([[0.2, 10.0, np.nan, np.nan]], shape=(1, 4)),
                    y=array([[0.0, 10.0, np.nan, np.nan]], shape=(1, 4)),
                    heading=array([[0.0, 0.0, np.nan, np.nan]], shape=(1, 4)),
                ),
                # Only 2 IDs (matching the valid columns)
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([3, 7]),
            ),
        ]

    @mark.parametrize(
        ["assignment", "states", "history", "ids", "expected"],
        [
            *cases(
                id_assignment=obstacles.numpy.id_assignment,
                position_extractor=NumPyObstaclePositionExtractor,
                data=data.numpy,
            ),
            *cases(
                id_assignment=obstacles.jax.id_assignment,
                position_extractor=JaxObstaclePositionExtractor,
                data=data.jax,
            ),
        ],
    )
    def test[ObstacleStatesForTimeStepT, IdT: ObstacleIds, HistoryT](
        self,
        assignment: ObstacleIdAssignment[ObstacleStatesForTimeStepT, IdT, HistoryT],
        states: ObstacleStatesForTimeStepT,
        history: HistoryT,
        ids: IdT,
        expected: IdT,
    ) -> None:
        assert np.allclose(assignment(states, history=history, ids=ids), expected)
