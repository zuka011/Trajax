from typing import Final

from faran_visualizer.api.simulation import SimulationVisualizer
from faran_visualizer.api.mpcc import MpccVisualizer


class visualizer:
    simulation: Final = SimulationVisualizer.create
    mpcc: Final = MpccVisualizer.create
