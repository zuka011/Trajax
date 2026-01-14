from typing import Final

from trajax_visualizer.api.simulation import SimulationVisualizer
from trajax_visualizer.api.mpcc import MpccVisualizer


class visualizer:
    simulation: Final = SimulationVisualizer.create
    mpcc: Final = MpccVisualizer.create
