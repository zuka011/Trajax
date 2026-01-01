import asyncio
from typing import Literal, cast, Final
from dataclasses import dataclass
from pathlib import Path

from aiopath import AsyncPath
from numtypes import Array, D, Dims, Dim1, Dim2, IndexArray

import orjson
import numpy as np

from tests.visualize.root import find_root


PROJECT_ROOT: Final = find_root()
VISUALIZATION_DIR: Final = PROJECT_ROOT / "tests" / "visualizations"
VISUALIZER_CLI: Final = PROJECT_ROOT / "visualizer" / "dist" / "cli" / "index.js"

DEFAULT_VEHICLE_WIDTH: Final = 1.2
DEFAULT_VEHICLE_WHEELBASE: Final = 2.5


type VehicleType = Literal["triangle", "car"]
type ObstacleCoordinate[T: int = int, K: int = int] = Array[Dims[T, K]]
type ObstacleForecast[T: int = int, H: int = int, K: int = int] = Array[Dims[T, H, K]]
type ObstacleForecastCovariance[T: int = int, H: int = int, K: int = int] = Array[
    Dims[T, H, D[2], D[2], K]
]


@dataclass(frozen=True)
class ReferenceTrajectory:
    x: Array[Dim1]
    y: Array[Dim1]


@dataclass(frozen=True)
class SimulationData:
    reference: ReferenceTrajectory
    positions_x: Array[Dim1]
    positions_y: Array[Dim1]
    headings: Array[Dim1]
    path_parameters: Array[Dim1]
    path_length: float
    time_step: float = 0.1
    errors: Array[Dim1] | None = None
    ghost_x: Array[Dim1] | None = None
    ghost_y: Array[Dim1] | None = None
    max_error: float = 1.0
    error_label: str | None = None
    vehicle_type: VehicleType = "triangle"
    wheelbase: float = DEFAULT_VEHICLE_WHEELBASE
    vehicle_width: float = DEFAULT_VEHICLE_WIDTH
    obstacle_positions_x: Array[Dim2] | None = None
    obstacle_positions_y: Array[Dim2] | None = None
    obstacle_headings: Array[Dim2] | None = None
    obstacle_forecast_x: ObstacleForecast | None = None
    obstacle_forecast_y: ObstacleForecast | None = None
    obstacle_forecast_heading: ObstacleForecast | None = None
    obstacle_forecast_covariance: ObstacleForecastCovariance | None = None

    def __post_init__(self) -> None:
        self._validate_time_step_count()

    @property
    def time_step_count(self) -> int:
        return len(self.positions_x)

    @property
    def time_steps(self) -> IndexArray[Dim1]:
        return np.arange(self.time_step_count)

    def _validate_time_step_count(self) -> None:
        for value, field in (
            (self.positions_x, "positions_x"),
            (self.positions_y, "positions_y"),
            (self.headings, "headings"),
            (self.path_parameters, "path_parameters"),
            (self.errors, "errors"),
            (self.ghost_x, "ghost_x"),
            (self.ghost_y, "ghost_y"),
            (self.obstacle_positions_x, "obstacle_positions_x"),
            (self.obstacle_positions_y, "obstacle_positions_y"),
            (self.obstacle_headings, "obstacle_headings"),
        ):
            assert value is None or len(value) == self.time_step_count, (
                f"Length of {field} does not match number of timesteps."
                f"Got {len(value)} values, expected {self.time_step_count}."
            )


class get:
    @staticmethod
    def errors(data: SimulationData) -> Array[Dim1]:
        if data.errors is None:
            return np.zeros(data.time_step_count)

        return data.errors

    @staticmethod
    def ghost_positions(data: SimulationData) -> tuple[Array[Dim1], Array[Dim1]]:
        if data.ghost_x is None or data.ghost_y is None:
            empty = np.empty((data.time_step_count,))
            return empty, empty

        return data.ghost_x, data.ghost_y

    @staticmethod
    def obstacle_positions(
        data: SimulationData,
    ) -> tuple[ObstacleCoordinate, ObstacleCoordinate, ObstacleCoordinate]:
        if data.obstacle_positions_x is None or data.obstacle_positions_y is None:
            empty = np.empty((data.time_step_count, 0))
            return empty, empty, empty

        obstacle_headings = (
            data.obstacle_headings
            if data.obstacle_headings is not None
            else np.zeros_like(data.obstacle_positions_x)
        )

        return data.obstacle_positions_x, data.obstacle_positions_y, obstacle_headings

    @staticmethod
    def obstacle_forecasts(
        data: SimulationData,
    ) -> tuple[ObstacleForecast, ObstacleForecast, ObstacleForecast]:
        if data.obstacle_forecast_x is None or data.obstacle_forecast_y is None:
            empty: ObstacleForecast = np.empty((data.time_step_count, 0, 0))
            return empty, empty, empty

        forecast_headings = (
            data.obstacle_forecast_heading
            if data.obstacle_forecast_heading is not None
            else np.zeros_like(data.obstacle_forecast_x)
        )

        return data.obstacle_forecast_x, data.obstacle_forecast_y, forecast_headings

    @staticmethod
    def obstacle_forecast_covariance(
        data: SimulationData,
    ) -> ObstacleForecastCovariance:
        if data.obstacle_forecast_covariance is None:
            empty = np.empty((data.time_step_count, 0, 2, 2, 0))
            return cast(ObstacleForecastCovariance, empty)

        return data.obstacle_forecast_covariance


def prepare(data: SimulationData) -> dict:
    errors = get.errors(data)
    ghost_x, ghost_y = get.ghost_positions(data)
    obstacle_x, obstacle_y, obstacle_heading = get.obstacle_positions(data)
    forecast_x, forecast_y, forecast_heading = get.obstacle_forecasts(data)
    forecast_covariance = get.obstacle_forecast_covariance(data)

    result: dict = {
        "reference": {"x": data.reference.x, "y": data.reference.y},
        "positions_x": data.positions_x,
        "positions_y": data.positions_y,
        "headings": data.headings,
        "path_parameters": data.path_parameters,
        "path_length": data.path_length,
        "time_step": data.time_step,
        "errors": errors,
        "ghost_x": ghost_x,
        "ghost_y": ghost_y,
        "max_error": data.max_error,
        "error_label": data.error_label,
        "vehicle_type": data.vehicle_type,
        "wheelbase": data.wheelbase,
        "vehicle_width": data.vehicle_width,
    }

    if obstacle_x.shape[1] > 0:
        result["obstacle_positions_x"] = obstacle_x
        result["obstacle_positions_y"] = obstacle_y
        result["obstacle_headings"] = obstacle_heading

    if forecast_x.shape[1] > 0 and forecast_x.shape[2] > 0:
        result["obstacle_forecast_x"] = forecast_x
        result["obstacle_forecast_y"] = forecast_y
        result["obstacle_forecast_heading"] = forecast_heading

    if forecast_covariance.shape[1] > 0 and forecast_covariance.shape[4] > 0:
        result["obstacle_forecast_covariance"] = forecast_covariance

    return result


def serialize(data: SimulationData) -> bytes:
    def handle_non_contiguous_arrays(obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist() if not obj.flags["C_CONTIGUOUS"] else obj

        raise TypeError(f"Cannot serialize {type(obj)}")

    return orjson.dumps(
        prepare(data),
        option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
        default=handle_non_contiguous_arrays,
    )


async def export(data: SimulationData, *, to: AsyncPath) -> None:
    serialized = await asyncio.get_running_loop().run_in_executor(
        None, lambda: serialize(data)
    )

    async with to.open("wb") as f:
        await f.write(serialized)


async def html(*, of: AsyncPath, to: AsyncPath, visualizer: AsyncPath) -> None:
    assert await visualizer.exists(), f"Visualizer CLI not found at {visualizer}."

    print(f"Generating HTML visualization from {of} to {to} using {visualizer}.")
    print(
        f"Visualizer command: {(command := ['node', str(visualizer), 'generate', str(of), '-o', str(to)])}"
    )

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

        await export(data, to=(json_data := self.output / f"{key}.json"))
        await html(
            of=json_data,
            to=(html_path := self.output / f"{key}.html"),
            visualizer=self.visualizer,
        )

        print(
            f"Simulation data saved to {json_data}. HTML visualization at {html_path}."
        )

    def can_visualize(self, data: object) -> bool:
        return isinstance(data, SimulationData)
