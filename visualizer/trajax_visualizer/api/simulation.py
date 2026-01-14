import asyncio
from typing import Literal, Sequence, Any
from dataclasses import dataclass
from pathlib import Path
from importlib.resources import files

from trajax_visualizer.api.config import config

from aiopath import AsyncPath
from numtypes import Array, D, Dims, Dim1

import msgspec
import numpy as np


def encode_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise NotImplementedError(f"Cannot serialize {type(obj)}")


encoder = msgspec.json.Encoder(enc_hook=encode_hook)


def cli_path() -> Path:
    return Path(str(files("trajax_visualizer") / "assets" / "cli.js"))


def default_output_directory() -> Path:
    return config.output_directory


class Types:
    type Vehicle = Literal["triangle", "car"]
    type Scale = Literal["linear", "log"]


class Arrays:
    type ObstacleCoordinates[T: int = int, K: int = int] = Array[Dims[T, K]]
    type ObstacleForecastCoordinates[T: int = int, H: int = int, K: int = int] = Array[
        Dims[T, H, K]
    ]
    type ObstacleForecastCovariances[T: int = int, H: int = int, K: int = int] = Array[
        Dims[T, H, D[2], D[2], K]
    ]
    type PlannedTrajectoryCoordinates[T: int = int, H: int = int] = Array[Dims[T, H]]


class Plot:
    class Series(msgspec.Struct, rename="camel", omit_defaults=True):
        label: str
        values: Array[Dim1]
        color: str | None = None

        @property
        def time_step_count(self) -> int:
            return len(self.values)

    class Bound(msgspec.Struct, rename="camel", omit_defaults=True):
        values: Array[Dim1] | float
        label: str | None = None

    class Band(msgspec.Struct, rename="camel", omit_defaults=True):
        lower: Array[Dim1]
        upper: Array[Dim1]
        color: str | None = None
        label: str | None = None

        def __post_init__(self) -> None:
            assert len(self.lower) == len(self.upper), (
                "Band lower and upper bounds must have the same length."
            )

        @property
        def time_step_count(self) -> int:
            return len(self.lower)

    class Additional(msgspec.Struct, rename="camel", omit_defaults=True):
        id: str
        name: str
        series: Sequence["Plot.Series"]
        y_axis_label: str
        upper_bound: "Plot.Bound | None" = None
        lower_bound: "Plot.Bound | None" = None
        bands: Sequence["Plot.Band"] | None = None
        y_axis_scale: Types.Scale | None = None
        group: str | None = None

        def __post_init__(self) -> None:
            assert len(self.series) > 0, (
                "Additional plot must have at least one series."
            )
            assert all(
                s.time_step_count == self.series[0].time_step_count for s in self.series
            ), "All series must have the same number of time steps."
            assert self.bands is None or all(
                b.time_step_count == self.series[0].time_step_count for b in self.bands
            ), "All bands must have the same number of time steps as series."

        @property
        def time_step_count(self) -> int:
            return self.series[0].time_step_count


class Visualizable:
    class ReferenceTrajectory(msgspec.Struct):
        x: Array[Dim1]
        y: Array[Dim1]

    class SimulationInfo(msgspec.Struct, rename="camel", omit_defaults=True):
        path_length: float
        time_step: float
        wheelbase: float | None = None
        vehicle_width: float | None = None
        vehicle_type: Types.Vehicle | None = None

    class EgoGhost(msgspec.Struct, rename="camel", omit_defaults=True):
        x: Array[Dim1]
        y: Array[Dim1]

        def __post_init__(self) -> None:
            assert len(self.x) == len(self.y), (
                f"Ego ghost x and y must have the same length. Got {len(self.x)} (x) and {len(self.y)} (y)."
            )

        @property
        def time_step_count(self) -> int:
            return len(self.x)

    class Ego(msgspec.Struct, rename="camel", omit_defaults=True):
        x: Array[Dim1]
        y: Array[Dim1]
        heading: Array[Dim1]
        path_parameter: Array[Dim1]
        ghost: "Visualizable.EgoGhost | None" = None

        def __post_init__(self) -> None:
            assert (
                len(self.x)
                == len(self.y)
                == len(self.heading)
                == len(self.path_parameter)
            ), (
                f"Ego x, y, heading, and path_parameter must have the same length. "
                f"Got {len(self.x)} (x), {len(self.y)} (y), {len(self.heading)} (heading), "
                f"and {len(self.path_parameter)} (path_parameter)."
            )
            assert (
                self.ghost is None or self.ghost.time_step_count == self.time_step_count
            ), (
                f"Ego ghost must have the same number of time steps as ego. "
                f"Got {self.ghost.time_step_count} (ghost) and {self.time_step_count} (ego)."
            )

        @property
        def time_step_count(self) -> int:
            return len(self.x)

    class PlannedTrajectory(msgspec.Struct, rename="camel", omit_defaults=True):
        x: Arrays.PlannedTrajectoryCoordinates
        y: Arrays.PlannedTrajectoryCoordinates

        def __post_init__(self) -> None:
            assert self.x.shape == self.y.shape, (
                f"Trajectory x and y must have the same shape. Got {self.x.shape} (x) and {self.y.shape} (y)."
            )

        @property
        def time_step_count(self) -> int:
            return len(self.x)

    class PlannedTrajectories(msgspec.Struct, rename="camel", omit_defaults=True):
        optimal: "Visualizable.PlannedTrajectory | None" = None
        nominal: "Visualizable.PlannedTrajectory | None" = None

        def __post_init__(self) -> None:
            assert (
                self.optimal is None
                or self.nominal is None
                or self.optimal.time_step_count == self.nominal.time_step_count
            ), (
                f"Optimal and nominal trajectories must have the same number of time steps. "
                f"Got {self.optimal.time_step_count} (optimal) and {self.nominal.time_step_count} (nominal)."
            )

        @property
        def time_step_count(self) -> int | None:
            if self.optimal is not None:
                return self.optimal.time_step_count

            if self.nominal is not None:
                return self.nominal.time_step_count

    class ObstacleForecast(msgspec.Struct, rename="camel", omit_defaults=True):
        x: Arrays.ObstacleForecastCoordinates
        y: Arrays.ObstacleForecastCoordinates
        heading: Arrays.ObstacleForecastCoordinates
        covariance: Arrays.ObstacleForecastCovariances | None = None

        def __post_init__(self) -> None:
            assert self.x.shape == self.y.shape == self.heading.shape, (
                f"Obstacle forecast x, y, and heading must have the same shape. "
                f"Got {self.x.shape} (x), {self.y.shape} (y), and {self.heading.shape} (heading)."
            )

            assert (
                self.covariance is None or self.covariance.shape[:2] == self.x.shape[:2]
            ), (
                f"Obstacle forecast covariance must have the same first two dimensions as x, y, and heading. "
                f"Got {self.covariance.shape} (covariance) and {self.x.shape} (x)."
            )

        @property
        def time_step_count(self) -> int:
            return self.x.shape[0]

    class Obstacles(msgspec.Struct, rename="camel", omit_defaults=True):
        x: Arrays.ObstacleCoordinates
        y: Arrays.ObstacleCoordinates
        heading: Arrays.ObstacleCoordinates
        forecast: "Visualizable.ObstacleForecast | None" = None

        def __post_init__(self) -> None:
            assert self.x.shape == self.y.shape == self.heading.shape, (
                f"Obstacle x, y, and heading must have the same shape. "
                f"Got {self.x.shape} (x), {self.y.shape} (y), and {self.heading.shape} (heading)."
            )

            assert (
                self.forecast is None
                or self.forecast.time_step_count == self.time_step_count
            ), (
                f"Obstacle forecast must have the same number of time steps as obstacles. "
                f"Got {self.forecast.time_step_count} (forecast) and {self.time_step_count} (obstacles)."
            )

        @property
        def time_step_count(self) -> int:
            return self.x.shape[0]

    class SimulationResult(msgspec.Struct, rename="camel", omit_defaults=True):
        info: "Visualizable.SimulationInfo"
        reference: "Visualizable.ReferenceTrajectory"
        ego: "Visualizable.Ego"
        trajectories: "Visualizable.PlannedTrajectories | None" = None
        obstacles: "Visualizable.Obstacles | None" = None
        additional_plots: Sequence[Plot.Additional] | None = None

        @staticmethod
        def create(
            *,
            info: "Visualizable.SimulationInfo",
            reference: "Visualizable.ReferenceTrajectory",
            ego: "Visualizable.Ego",
            trajectories: "Visualizable.PlannedTrajectories | None" = None,
            obstacles: "Visualizable.Obstacles | None" = None,
            additional_plots: Sequence[Plot.Additional] | None = None,
        ) -> "Visualizable.SimulationResult":
            return Visualizable.SimulationResult(
                info=info,
                reference=reference,
                ego=ego,
                trajectories=trajectories,
                obstacles=obstacles,
                additional_plots=additional_plots,
            )

        def __post_init__(self) -> None:
            for value, name in (
                (self.ego, "ego"),
                (self.trajectories, "trajectories"),
                (self.obstacles, "obstacles"),
            ):
                assert value is None or value.time_step_count == self.time_step_count, (
                    f"{name} length ({value.time_step_count}) != time step count ({self.time_step_count})"
                )

            if self.additional_plots:
                for plot in self.additional_plots:
                    assert plot.time_step_count == self.time_step_count, (
                        f"Plot '{plot.name}' has {plot.time_step_count} steps, expected {self.time_step_count}"
                    )

        @property
        def time_step_count(self) -> int:
            return self.ego.time_step_count


def serialize(data: Visualizable.SimulationResult) -> bytes:
    return encoder.encode(data)


async def export(data: Visualizable.SimulationResult, *, to: AsyncPath) -> None:
    serialized = await asyncio.get_running_loop().run_in_executor(
        None, lambda: serialize(data)
    )

    async with to.open("wb") as f:
        await f.write(serialized)


async def html(*, of: AsyncPath, to: AsyncPath, visualizer: AsyncPath) -> None:
    assert await visualizer.exists(), (
        f"Visualizer CLI not found at {visualizer}. "
        "Make sure Node.js is installed and the CLI has been built."
    )

    command = ["node", str(visualizer), "generate", str(of), "-o", str(to)]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Visualization generation failed: {stderr.decode()}")


@dataclass(kw_only=True, frozen=True)
class SimulationVisualizer:
    output: AsyncPath
    visualizer: AsyncPath

    @staticmethod
    def create(
        *,
        output: str = "mpcc-simulation",
        output_directory: str | None = None,
    ) -> "SimulationVisualizer":
        directory = (
            Path(output_directory)
            if output_directory is not None
            else default_output_directory()
        )
        return SimulationVisualizer(
            output=AsyncPath(directory) / output,
            visualizer=AsyncPath(cli_path()),
        )

    async def __call__(self, data: Visualizable.SimulationResult, *, key: str) -> None:
        await self.output.mkdir(parents=True, exist_ok=True)

        await export(data, to=(json_path := self.output / f"{key}.json"))
        await html(
            of=json_path,
            to=(html_path := self.output / f"{key}.html"),
            visualizer=self.visualizer,
        )

        print(f"Saved: {json_path}, {html_path}")

    def can_visualize(self, data: object) -> bool:
        return isinstance(data, Visualizable.SimulationResult)
