from typing import Final

from trajax.trajectory.basic import NumpyLineTrajectory
from trajax.trajectory.accelerated import JaxLineTrajectory


class trajectory:
    class numpy:
        line: Final = NumpyLineTrajectory.create

    class jax:
        line: Final = JaxLineTrajectory.create
