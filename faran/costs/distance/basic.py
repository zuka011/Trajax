import numpy as np

from faran.types import NumPySampledObstaclePositions, NumPySampledObstacleHeadings


def replace_missing[T: int, K: int, N: int](
    *,
    positions: NumPySampledObstaclePositions[T, K, N],
    headings: NumPySampledObstacleHeadings[T, K, N],
) -> tuple[
    NumPySampledObstaclePositions[T, K, N], NumPySampledObstacleHeadings[T, K, N]
]:
    return (
        NumPySampledObstaclePositions.create(
            x=np.nan_to_num(positions.x(), nan=np.inf, posinf=np.inf, neginf=np.inf),
            y=np.nan_to_num(positions.y(), nan=np.inf, posinf=np.inf, neginf=np.inf),
        ),
        NumPySampledObstacleHeadings.create(
            heading=np.nan_to_num(headings.heading(), nan=0.0, posinf=0.0, neginf=0.0)
        ),
    )
