import numpy as np

from trajax.obstacles import NumPySampledObstacleStates


def replace_missing[T: int, K: int, N: int](
    obstacle_states: NumPySampledObstacleStates[T, K, N],
) -> NumPySampledObstacleStates[T, K, N]:
    return NumPySampledObstacleStates.create(
        x=np.nan_to_num(obstacle_states.x(), nan=np.inf, posinf=np.inf, neginf=np.inf),
        y=np.nan_to_num(obstacle_states.y(), nan=np.inf, posinf=np.inf, neginf=np.inf),
        heading=np.nan_to_num(
            obstacle_states.heading(), nan=0.0, posinf=0.0, neginf=0.0
        ),
    )
