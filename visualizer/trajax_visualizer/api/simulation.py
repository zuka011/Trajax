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


type VehicleType = Literal["triangle", "car"]
type ScaleType = Literal["linear", "log"]
type ObstacleCoordinateArray[T: int = int, K: int = int] = Array[Dims[T, K]]
type ObstacleForecastArray[T: int = int, H: int = int, K: int = int] = Array[
    Dims[T, H, K]
]
type ObstacleForecastCovarianceArray[T: int = int, H: int = int, K: int = int] = Array[
    Dims[T, H, D[2], D[2], K]
]
type PlannedTrajectoryCoordinateArray[T: int = int, H: int = int] = Array[Dims[T, H]]


class ReferenceTrajectory(msgspec.Struct):
    x: Array[Dim1]
    y: Array[Dim1]


class SimulationInfo(msgspec.Struct, rename="camel", omit_defaults=True):
    path_length: float
    time_step: float | None = None
    wheelbase: float | None = None
    vehicle_width: float | None = None
    vehicle_type: VehicleType | None = None


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
    ghost: EgoGhost | None = None

    def __post_init__(self) -> None:
        assert (
            len(self.x) == len(self.y) == len(self.heading) == len(self.path_parameter)
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
    x: PlannedTrajectoryCoordinateArray
    y: PlannedTrajectoryCoordinateArray

    def __post_init__(self) -> None:
        assert self.x.shape == self.y.shape, (
            f"Trajectory x and y must have the same shape. Got {self.x.shape} (x) and {self.y.shape} (y)."
        )

    @property
    def time_step_count(self) -> int:
        return len(self.x)


class PlannedTrajectories(msgspec.Struct, rename="camel", omit_defaults=True):
    optimal: PlannedTrajectory | None = None
    nominal: PlannedTrajectory | None = None

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
    x: ObstacleForecastArray
    y: ObstacleForecastArray
    heading: ObstacleForecastArray
    covariance: ObstacleForecastCovarianceArray | None = None

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
    x: ObstacleCoordinateArray
    y: ObstacleCoordinateArray
    heading: ObstacleCoordinateArray
    forecast: ObstacleForecast | None = None

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


class PlotSeries(msgspec.Struct, rename="camel", omit_defaults=True):
    label: str
    values: Array[Dim1]
    color: str | None = None

    @property
    def time_step_count(self) -> int:
        return len(self.values)


class PlotBound(msgspec.Struct, rename="camel", omit_defaults=True):
    values: Array[Dim1] | float
    label: str | None = None


class PlotBand(msgspec.Struct, rename="camel", omit_defaults=True):
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


class AdditionalPlot(msgspec.Struct, rename="camel", omit_defaults=True):
    id: str
    name: str
    series: Sequence[PlotSeries]
    y_axis_label: str
    upper_bound: PlotBound | None = None
    lower_bound: PlotBound | None = None
    bands: Sequence[PlotBand] | None = None
    y_axis_scale: ScaleType | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        assert len(self.series) > 0, "Additional plot must have at least one series."
        assert all(
            s.time_step_count == self.series[0].time_step_count for s in self.series
        ), "All series must have the same number of time steps."
        assert self.bands is None or all(
            b.time_step_count == self.series[0].time_step_count for b in self.bands
        ), "All bands must have the same number of time steps as series."

    @property
    def time_step_count(self) -> int:
        return self.series[0].time_step_count


class SimulationData(msgspec.Struct, rename="camel", omit_defaults=True):
    info: SimulationInfo
    reference: ReferenceTrajectory
    ego: Ego
    trajectories: PlannedTrajectories | None = None
    obstacles: Obstacles | None = None
    additional_plots: Sequence[AdditionalPlot] | None = None

    @staticmethod
    def create(
        *,
        info: SimulationInfo,
        reference: ReferenceTrajectory,
        ego: Ego,
        trajectories: PlannedTrajectories | None = None,
        obstacles: Obstacles | None = None,
        additional_plots: Sequence[AdditionalPlot] | None = None,
    ) -> "SimulationData":
        return SimulationData(
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


def serialize(data: SimulationData) -> bytes:
    return encoder.encode(data)


async def export(data: SimulationData, *, to: AsyncPath) -> None:
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

    async def __call__(self, data: SimulationData, *, key: str) -> None:
        await self.output.mkdir(parents=True, exist_ok=True)

        await export(data, to=(json_path := self.output / f"{key}.json"))
        await html(
            of=json_path,
            to=(html_path := self.output / f"{key}.html"),
            visualizer=self.visualizer,
        )

        print(f"Saved: {json_path}, {html_path}")

    def can_visualize(self, data: object) -> bool:
        return isinstance(data, SimulationData)
