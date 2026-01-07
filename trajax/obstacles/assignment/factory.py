from typing import Final

from trajax.obstacles.assignment.basic import NumPyHungarianObstacleIdAssignment
from trajax.obstacles.assignment.accelerated import JaxHungarianObstacleIdAssignment


class id_assignment:
    class numpy:
        hungarian: Final = NumPyHungarianObstacleIdAssignment.create

    class jax:
        hungarian: Final = JaxHungarianObstacleIdAssignment.create
