from typing import Final

from trajax.obstacles.assignment.basic import NumPyHungarianObstacleIdAssignment


class id_assignment:
    class numpy:
        hungarian: Final = NumPyHungarianObstacleIdAssignment.create
