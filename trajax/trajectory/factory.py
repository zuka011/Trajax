from typing import Final

from trajax.trajectory.line import NumPyLineTrajectory, JaxLineTrajectory
from trajax.trajectory.waypoints import NumPyWaypointsTrajectory, JaxWaypointsTrajectory


class trajectory:
    class numpy:
        line: Final = NumPyLineTrajectory.create
        waypoints: Final = NumPyWaypointsTrajectory.create

    class jax:
        line: Final = JaxLineTrajectory.create
        waypoints: Final = JaxWaypointsTrajectory.create
