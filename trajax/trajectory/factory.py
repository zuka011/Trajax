from typing import Final

from trajax.trajectory.line import NumpyLineTrajectory, JaxLineTrajectory
from trajax.trajectory.waypoints import NumpyWaypointsTrajectory, JaxWaypointsTrajectory


class trajectory:
    class numpy:
        line: Final = NumpyLineTrajectory.create
        waypoints: Final = NumpyWaypointsTrajectory.create

    class jax:
        line: Final = JaxLineTrajectory.create
        waypoints: Final = JaxWaypointsTrajectory.create
