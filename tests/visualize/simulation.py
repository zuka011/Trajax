import asyncio
from typing import Literal, Final, Sequence, Any
from dataclasses import dataclass
from pathlib import Path

from aiopath import AsyncPath
from numtypes import Array, D, Dims, Dim1, IndexArray

import msgspec
import numpy as np

from tests.visualize.root import find_root


def enc_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise NotImplementedError(f"Cannot serialize {type(obj)}")


encoder = msgspec.json.Encoder(enc_hook=enc_hook)


PROJECT_ROOT: Final = find_root()
VISUALIZATION_DIR: Final = PROJECT_ROOT / "tests" / "visualizations"
VISUALIZER_CLI: Final = PROJECT_ROOT / "visualizer" / "dist" / "cli" / "index.js"

type VehicleType = Literal["triangle", "car"]
type ScaleType = Literal["linear", "log"]
type ObstacleCoordinate[T: int = int, K: int = int] = Array[Dims[T, K]]
type ObstacleForecast[T: int = int, H: int = int, K: int = int] = Array[Dims[T, H, K]]
type ObstacleForecastCovariance[T: int = int, H: int = int, K: int = int] = Array[
    Dims[T, H, D[2], D[2], K]
]
type PlannedTrajectory[T: int = int, H: int = int] = Array[Dims[T, H]]


class ReferenceTrajectory(msgspec.Struct):
    x: Array[Dim1]
    y: Array[Dim1]


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
    reference: ReferenceTrajectory
    positions_x: Array[Dim1]
    positions_y: Array[Dim1]
    headings: Array[Dim1]
    path_parameters: Array[Dim1]
    path_length: float
    time_step: float | None = None
    wheelbase: float | None = None
    vehicle_width: float | None = None
    ghost_x: Array[Dim1] | None = None
    ghost_y: Array[Dim1] | None = None
    optimal_trajectory_x: PlannedTrajectory | None = None
    optimal_trajectory_y: PlannedTrajectory | None = None
    nominal_trajectory_x: PlannedTrajectory | None = None
    nominal_trajectory_y: PlannedTrajectory | None = None
    vehicle_type: VehicleType | None = None
    obstacle_positions_x: ObstacleCoordinate | None = None
    obstacle_positions_y: ObstacleCoordinate | None = None
    obstacle_headings: ObstacleCoordinate | None = None
    obstacle_forecast_x: ObstacleForecast | None = None
    obstacle_forecast_y: ObstacleForecast | None = None
    obstacle_forecast_heading: ObstacleForecast | None = None
    obstacle_forecast_covariance: ObstacleForecastCovariance | None = None
    additional_plots: Sequence[AdditionalPlot] | None = None

    @staticmethod
    def create(
        *,
        reference: ReferenceTrajectory,
        positions_x: Array[Dim1],
        positions_y: Array[Dim1],
        headings: Array[Dim1],
        path_parameters: Array[Dim1],
        path_length: float,
        time_step: float | None = None,
        ghost_x: Array[Dim1] | None = None,
        ghost_y: Array[Dim1] | None = None,
        optimal_trajectory_x: PlannedTrajectory | None = None,
        optimal_trajectory_y: PlannedTrajectory | None = None,
        nominal_trajectory_x: PlannedTrajectory | None = None,
        nominal_trajectory_y: PlannedTrajectory | None = None,
        vehicle_type: VehicleType | None = None,
        wheelbase: float | None = None,
        vehicle_width: float | None = None,
        obstacle_positions_x: ObstacleCoordinate | None = None,
        obstacle_positions_y: ObstacleCoordinate | None = None,
        obstacle_headings: ObstacleCoordinate | None = None,
        obstacle_forecast_x: ObstacleForecast | None = None,
        obstacle_forecast_y: ObstacleForecast | None = None,
        obstacle_forecast_heading: ObstacleForecast | None = None,
        obstacle_forecast_covariance: ObstacleForecastCovariance | None = None,
        additional_plots: Sequence[AdditionalPlot] | None = None,
    ) -> "SimulationData":
        return SimulationData(
            reference=reference,
            positions_x=positions_x,
            positions_y=positions_y,
            headings=headings,
            path_parameters=path_parameters,
            path_length=path_length,
            time_step=time_step,
            ghost_x=ghost_x,
            ghost_y=ghost_y,
            optimal_trajectory_x=optimal_trajectory_x,
            optimal_trajectory_y=optimal_trajectory_y,
            nominal_trajectory_x=nominal_trajectory_x,
            nominal_trajectory_y=nominal_trajectory_y,
            vehicle_type=vehicle_type,
            wheelbase=wheelbase,
            vehicle_width=vehicle_width,
            obstacle_positions_x=obstacle_positions_x,
            obstacle_positions_y=obstacle_positions_y,
            obstacle_headings=obstacle_headings,
            obstacle_forecast_x=obstacle_forecast_x,
            obstacle_forecast_y=obstacle_forecast_y,
            obstacle_forecast_heading=obstacle_forecast_heading,
            obstacle_forecast_covariance=obstacle_forecast_covariance,
            additional_plots=additional_plots,
        )

    def __post_init__(self) -> None:
        self._validate_time_step_counts()

    @property
    def time_step_count(self) -> int:
        return len(self.positions_x)

    @property
    def time_steps(self) -> IndexArray[Dim1]:
        return np.arange(self.time_step_count)

    def _validate_time_step_counts(self) -> None:
        for value, name in (
            (self.positions_x, "positions_x"),
            (self.positions_y, "positions_y"),
            (self.headings, "headings"),
            (self.path_parameters, "path_parameters"),
            (self.ghost_x, "ghost_x"),
            (self.ghost_y, "ghost_y"),
            (self.optimal_trajectory_x, "optimal_trajectory_x"),
            (self.optimal_trajectory_y, "optimal_trajectory_y"),
            (self.nominal_trajectory_x, "nominal_trajectory_x"),
            (self.nominal_trajectory_y, "nominal_trajectory_y"),
            (self.obstacle_positions_x, "obstacle_positions_x"),
            (self.obstacle_positions_y, "obstacle_positions_y"),
            (self.obstacle_headings, "obstacle_headings"),
            (self.obstacle_forecast_x, "obstacle_forecast_x"),
            (self.obstacle_forecast_y, "obstacle_forecast_y"),
            (self.obstacle_forecast_heading, "obstacle_forecast_heading"),
            (self.obstacle_forecast_covariance, "obstacle_forecast_covariance"),
        ):
            assert value is None or len(value) == self.time_step_count, (
                f"{name} length ({len(value)}) != time step count ({self.time_step_count})"
            )

        if self.additional_plots:
            for plot in self.additional_plots:
                assert plot.time_step_count == self.time_step_count, (
                    f"Plot '{plot.name}' has {plot.time_step_count} steps, expected {self.time_step_count}"
                )


def serialize(data: SimulationData) -> bytes:
    return encoder.encode(data)


async def export(data: SimulationData, *, to: AsyncPath) -> None:
    serialized = await asyncio.get_running_loop().run_in_executor(
        None, lambda: serialize(data)
    )

    async with to.open("wb") as f:
        await f.write(serialized)


async def html(*, of: AsyncPath, to: AsyncPath, visualizer: AsyncPath) -> None:
    assert await visualizer.exists(), f"Visualizer CLI not found at {visualizer}."

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
        *, output: str = "mpcc-simulation", directory: Path | None = None
    ) -> "SimulationVisualizer":
        return SimulationVisualizer(
            output=AsyncPath(directory if directory is not None else VISUALIZATION_DIR)
            / output,
            visualizer=AsyncPath(VISUALIZER_CLI),
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
