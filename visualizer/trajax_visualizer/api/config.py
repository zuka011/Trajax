from typing import Final
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisualizerConfig:
    """Global configuration for the visualizer."""

    output_directory: Path = Path.cwd() / "visualizations"

    def configure(self, *, output_directory: str | Path) -> None:
        """Configure the visualizer globally.

        Args:
            output_directory: The directory where visualizations will be saved.
        """
        self.output_directory = Path(output_directory)


config: Final = VisualizerConfig()
configure: Final = config.configure
