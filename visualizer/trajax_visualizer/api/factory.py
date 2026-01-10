from typing import Final

from trajax_visualizer.api.mpcc import MpccVisualizer


class visualizer:
    mpcc: Final = MpccVisualizer.create
