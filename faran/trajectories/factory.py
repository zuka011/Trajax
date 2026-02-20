from typing import Final

from faran.trajectories.line import NumPyLineTrajectory, JaxLineTrajectory
from faran.trajectories.waypoints import (
    NumPyWaypointsTrajectory,
    JaxWaypointsTrajectory,
)


class trajectory:
    """Factory namespace for creating reference trajectories."""

    class numpy:
        line: Final = NumPyLineTrajectory.create
        waypoints: Final = NumPyWaypointsTrajectory.create

    class jax:
        line: Final = JaxLineTrajectory.create
        waypoints: Final = JaxWaypointsTrajectory.create
