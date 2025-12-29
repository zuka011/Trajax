from typing import Literal
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numtypes import Array, Dims, Dim1, Dim2, IndexArray
from aiopath import AsyncPath

from bokeh.plotting import figure
from bokeh.models.plots import Plot as Figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource, Slider, CustomJS, Div, Button, Column
from bokeh.layouts import column, row
from bokeh.events import DocumentReady


type VehicleType = Literal["triangle", "car"]
type ObstacleCoordinate[T: int = int, K: int = int] = Array[Dims[T, K]]
type ObstacleForecast[T: int = int, H: int = int, K: int = int] = Array[Dims[T, H, K]]


class Theme:
    """Centralized styling constants."""

    class Colors:
        PRIMARY = "#3498db"
        SECONDARY = "#2ecc71"
        ACCENT = "#e74c3c"
        ACCENT_DARK = "#c0392b"
        REFERENCE = "#bdc3c7"
        BACKGROUND = "#fafafa"
        BORDER = "#ffffff"
        TEXT = "#2c3e50"
        INFO_BG = "#f8f9fa"
        INFO_BORDER = "#dee2e6"

    class Sizes:
        VEHICLE_SIZE = 15
        LINE_WIDTH = 3
        MARKER_SIZE = 10
        DEFAULT_VEHICLE_WIDTH = 1.2
        DEFAULT_VEHICLE_WHEELBASE = 2.5

    class Constants:
        FRAME_INTERVAL_MS = 50


INFO_TEMPLATE = """
    <style>
        .bk-clearfix {{ width: 100% !important; }}
    </style>
    <div style="background: {bg}; border: 1px solid {border}; 
            border-radius: 8px; padding: 15px; font-family: system-ui, sans-serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 100%;">
        <h3 style="margin: 0 0 12px 0; color: {text_color}; font-size: 14px;">
            Simulation State
        </h3>
        <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
            <tr><td style="padding: 4px 0; color: #666;">Time</td>
                <td style="text-align: right; font-weight: 500;">${{time_s}} s</td></tr>
            <tr><td style="padding: 4px 0; color: #666;">Position</td>
                <td style="text-align: right; font-weight: 500;">(${{pos_x}}, ${{pos_y}}) m</td></tr>
            <tr><td style="padding: 4px 0; color: #666;">Heading</td>
                <td style="text-align: right; font-weight: 500;">${{heading}}°</td></tr>
            <tr><td style="padding: 4px 0; color: #666;">Path Parameter</td>
                <td style="text-align: right; font-weight: 500;">${{path_param}} / ${{path_length}} m</td></tr>
            <tr><td style="padding: 4px 0; color: #666;">Progress</td>
                <td style="text-align: right; font-weight: 500;">${{progress}}%</td></tr>
        </table>
    </div>
""".format(
    bg=Theme.Colors.INFO_BG,
    border=Theme.Colors.INFO_BORDER,
    text_color=Theme.Colors.TEXT,
)

INFO_UPDATE_CODE = """
    const t = slider.value;
    const time_s = (t * dt).toFixed(2);
    const heading_deg = (headings[t] * 180 / Math.PI).toFixed(1);
    const progress_pct = (100 * path_parameters[t] / path_length).toFixed(1);

    info_div.text = template
        .replace('${time_s}', time_s)
        .replace('${pos_x}', positions_x[t].toFixed(2))
        .replace('${pos_y}', positions_y[t].toFixed(2))
        .replace('${heading}', heading_deg)
        .replace('${path_param}', path_parameters[t].toFixed(2))
        .replace('${path_length}', path_length.toFixed(1))
        .replace('${progress}', progress_pct);
"""


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
    error_label: str = "Lateral Error"
    vehicle_type: VehicleType = "triangle"
    wheelbase: float = Theme.Sizes.DEFAULT_VEHICLE_WHEELBASE
    vehicle_width: float = Theme.Sizes.DEFAULT_VEHICLE_WIDTH
    obstacle_positions_x: Array[Dim2] | None = None
    obstacle_positions_y: Array[Dim2] | None = None
    obstacle_headings: Array[Dim2] | None = None
    obstacle_forecast_x: ObstacleForecast | None = None
    obstacle_forecast_y: ObstacleForecast | None = None
    obstacle_forecast_heading: ObstacleForecast | None = None

    def __post_init__(self) -> None:
        assert len(self.positions_x) == self.time_step_count, (
            f"Length of positions_x does not match number of timesteps. "
            f"Got {len(self.positions_x)} positions, expected {self.time_step_count}."
        )
        assert len(self.positions_y) == self.time_step_count, (
            f"Length of positions_y does not match number of timesteps."
            f"Got {len(self.positions_y)} positions, expected {self.time_step_count}."
        )
        assert len(self.headings) == self.time_step_count, (
            f"Length of headings does not match number of timesteps."
            f"Got {len(self.headings)} headings, expected {self.time_step_count}."
        )
        assert len(self.path_parameters) == self.time_step_count, (
            f"Length of path_parameters does not match number of timesteps."
            f"Got {len(self.path_parameters)} parameters, expected {self.time_step_count}."
        )
        assert self.errors is None or len(self.errors) == self.time_step_count, (
            f"Length of errors does not match number of timesteps."
            f"Got {len(self.errors)} errors, expected {self.time_step_count}."
        )
        assert self.ghost_x is None or len(self.ghost_x) == self.time_step_count, (
            f"Length of ghost_x does not match number of timesteps."
            f"Got {len(self.ghost_x)} positions, expected {self.time_step_count}."
        )
        assert self.ghost_y is None or len(self.ghost_y) == self.time_step_count, (
            f"Length of ghost_y does not match number of timesteps."
            f"Got {len(self.ghost_y)} positions, expected {self.time_step_count}."
        )
        assert (
            self.obstacle_positions_x is None
            or len(self.obstacle_positions_x) == self.time_step_count
        ), (
            f"Length of obstacle_positions_x does not match number of timesteps."
            f"Got {len(self.obstacle_positions_x)} positions, expected {self.time_step_count}."
        )
        assert (
            self.obstacle_positions_y is None
            or len(self.obstacle_positions_y) == self.time_step_count
        ), (
            f"Length of obstacle_positions_y does not match number of timesteps."
            f"Got {len(self.obstacle_positions_y)} positions, expected {self.time_step_count}."
        )
        assert (
            self.obstacle_headings is None
            or len(self.obstacle_headings) == self.time_step_count
        ), (
            f"Length of obstacle_headings does not match number of timesteps."
            f"Got {len(self.obstacle_headings)} headings, expected {self.time_step_count}."
        )

    @property
    def time_step_count(self) -> int:
        return len(self.positions_x)

    @property
    def time_steps(self) -> IndexArray[Dim1]:
        return np.arange(self.time_step_count)


@dataclass(kw_only=True, frozen=True)
class SimulationVisualizer:
    """Generic visualizer for trajectory-following simulations using Bokeh."""

    output: AsyncPath

    @staticmethod
    def create(
        *, output: str = "simulation", directory: Path | None = None
    ) -> "SimulationVisualizer":
        return SimulationVisualizer(
            output=AsyncPath(
                directory
                if directory is not None
                else Path(__file__).parent.parent / "visualizations"
            )
            / output
        )

    async def __call__(self, data: SimulationData, *, key: str) -> None:
        await self.output.mkdir(parents=True, exist_ok=True)
        await create_animation_html(data, output := self.output / f"{key}.html")

        print(f"\nVisualization saved to: {output}")

    def can_visualize(self, data: object) -> bool:
        return isinstance(data, SimulationData)


async def create_animation_html(data: SimulationData, output: AsyncPath) -> None:
    sources = create_data_sources(data)

    trajectory_plot = create_trajectory_plot(data)
    progress_plot = create_progress_plot(data)
    error_plot = create_error_plot(data)

    add_vehicle_to_plot(trajectory_plot, sources["vehicle"], data)
    add_trajectory_trace_to_plot(trajectory_plot, sources["trajectory"])
    add_ghost_vehicle_to_plot(trajectory_plot, sources["ghost"])
    add_obstacles_to_plot(trajectory_plot, sources["obstacles"], data)
    add_obstacle_forecasts_to_plot(trajectory_plot, sources["forecast"], data)
    add_progress_marker_to_plot(progress_plot, sources["progress"])
    add_error_marker_to_plot(error_plot, sources["error"])

    time_slider = create_time_slider(data.time_step_count)
    play_button = create_play_button(time_slider, data.time_step_count)
    speed_selector = create_speed_selector()
    reset_button = create_reset_button(time_slider)

    info_section = info_section_placeholder()

    slider_callback = create_slider_callback(sources=sources, data=data)
    time_slider.js_on_change("value", slider_callback)

    setup_info_updates(info_section, time_slider, data, trajectory_plot)

    layout = create_layout(
        trajectory_plot=trajectory_plot,
        progress_plot=progress_plot,
        error_plot=error_plot,
        info_section=info_section,
        time_slider=time_slider,
        play_button=play_button,
        speed_selector=speed_selector,
        reset_button=reset_button,
    )

    await save(layout, to=output)


def create_data_sources(data: SimulationData) -> dict[str, ColumnDataSource]:
    errors = get_errors(data)
    ghost_x, ghost_y = get_ghost_positions(data)
    obstacle_x, obstacle_y, obstacle_heading = get_obstacle_positions(data)
    forecast_x, forecast_y, forecast_heading = get_obstacle_forecasts(data)

    return {
        "vehicle": ColumnDataSource(
            data={
                "x": [data.positions_x[0]],
                "y": [data.positions_y[0]],
                "heading": [data.headings[0]],
            }
        ),
        "trajectory": ColumnDataSource(
            data={
                "x": data.positions_x[:1].tolist(),
                "y": data.positions_y[:1].tolist(),
            }
        ),
        "progress": ColumnDataSource(
            data={
                "timestep": [0],
                "path_parameter": [data.path_parameters[0]],
            }
        ),
        "ghost": ColumnDataSource(
            data={
                "x": [ghost_x[0]],
                "y": [ghost_y[0]],
            }
        ),
        "error": ColumnDataSource(
            data={
                "timestep": [0],
                "error": [errors[0]],
            }
        ),
        "obstacles": ColumnDataSource(
            data={
                "x": obstacle_x[0].tolist(),
                "y": obstacle_y[0].tolist(),
                "heading": obstacle_heading[0].tolist(),
            }
        ),
        "forecast": ColumnDataSource(
            data={
                "xs": (
                    forecasts := extract_forecasts(
                        forecast_x, forecast_y, forecast_heading, timestep=0
                    )
                )[0],
                "ys": forecasts[1],
                "headings": forecasts[2],
                "arrow_x": (
                    arrows := extract_forecast_arrows(
                        forecast_x, forecast_y, forecast_heading, timestep=0
                    )
                )[0],
                "arrow_y": arrows[1],
                "arrow_heading": arrows[2],
            }
        ),
        "simulation": ColumnDataSource(
            data={
                "positions_x": data.positions_x.tolist(),
                "positions_y": data.positions_y.tolist(),
                "headings": data.headings.tolist(),
                "path_parameters": data.path_parameters.tolist(),
                "errors": errors.tolist(),
                "ghost_x": ghost_x.tolist(),
                "ghost_y": ghost_y.tolist(),
                "obstacle_positions_x": obstacle_x.tolist(),
                "obstacle_positions_y": obstacle_y.tolist(),
                "obstacle_headings": obstacle_heading.tolist(),
                "obstacle_forecast_x": forecast_x.tolist(),
                "obstacle_forecast_y": forecast_y.tolist(),
                "obstacle_forecast_headings": forecast_heading.tolist(),
            }
        ),
        "reference": ColumnDataSource(
            data={
                "x": data.reference.x.tolist(),
                "y": data.reference.y.tolist(),
            }
        ),
    }


def get_errors(data: SimulationData) -> Array[Dim1]:
    if data.errors is not None:
        return data.errors

    return compute_lateral_errors(data)


def get_ghost_positions(data: SimulationData) -> tuple[Array[Dim1], Array[Dim1]]:
    if data.ghost_x is not None and data.ghost_y is not None:
        return data.ghost_x, data.ghost_y

    return compute_ghost_positions_from_closest_point(data)


def get_obstacle_positions(
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


def get_obstacle_forecasts(
    data: SimulationData,
) -> tuple[ObstacleForecast, ObstacleForecast, ObstacleForecast]:
    if data.obstacle_forecast_x is None or data.obstacle_forecast_y is None:
        empty: ObstacleForecast = np.empty((data.time_step_count, 0, 0))
        return empty, empty, empty

    forecast_headings = (
        # NOTE: -pi/2 to compensate for default arrow orientation in Bokeh
        data.obstacle_forecast_heading - (np.pi / 2)
        if data.obstacle_forecast_heading is not None
        else np.zeros_like(data.obstacle_forecast_x)
    )

    return data.obstacle_forecast_x, data.obstacle_forecast_y, forecast_headings


def extract_forecasts(
    forecast_x: ObstacleForecast,
    forecast_y: ObstacleForecast,
    forecast_heading: ObstacleForecast,
    *,
    timestep: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    if forecast_x.shape[1] == 0 or forecast_x.shape[2] == 0:
        return [], [], []

    obstacle_count = forecast_x.shape[2]

    xs = [forecast_x[timestep, :, k].tolist() for k in range(obstacle_count)]
    ys = [forecast_y[timestep, :, k].tolist() for k in range(obstacle_count)]
    headings = [
        forecast_heading[timestep, :, k].tolist() for k in range(obstacle_count)
    ]

    return xs, ys, headings


def extract_forecast_arrows(
    forecast_x: ObstacleForecast,
    forecast_y: ObstacleForecast,
    forecast_heading: ObstacleForecast,
    *,
    timestep: int,
) -> tuple[list[float], list[float], list[float]]:
    if forecast_x.shape[1] == 0 or forecast_x.shape[2] == 0:
        return [], [], []

    obstacle_count = forecast_x.shape[2]
    last_idx = forecast_x.shape[1] - 1

    arrow_x = [float(forecast_x[timestep, last_idx, k]) for k in range(obstacle_count)]
    arrow_y = [float(forecast_y[timestep, last_idx, k]) for k in range(obstacle_count)]
    arrow_heading = [
        float(forecast_heading[timestep, last_idx, k]) for k in range(obstacle_count)
    ]

    return arrow_x, arrow_y, arrow_heading


def compute_lateral_errors(data: SimulationData) -> Array[Dim1]:
    errors = np.zeros(data.time_step_count)

    for i in range(data.time_step_count):
        distances = np.sqrt(
            (data.reference.x - data.positions_x[i]) ** 2
            + (data.reference.y - data.positions_y[i]) ** 2
        )
        errors[i] = np.min(distances)

    return errors


def compute_ghost_positions_from_closest_point(
    data: SimulationData,
) -> tuple[Array[Dim1], Array[Dim1]]:
    ghost_x = np.zeros(data.time_step_count)
    ghost_y = np.zeros(data.time_step_count)

    for i in range(data.time_step_count):
        distances = np.sqrt(
            (data.reference.x - data.positions_x[i]) ** 2
            + (data.reference.y - data.positions_y[i]) ** 2
        )
        closest_idx = np.argmin(distances)
        ghost_x[i] = data.reference.x[closest_idx]
        ghost_y[i] = data.reference.y[closest_idx]

    return ghost_x, ghost_y


def create_trajectory_plot(data: SimulationData) -> Figure:
    plot = figure(
        title="Vehicle Trajectory",
        x_axis_label="X Position (m)",
        y_axis_label="Y Position (m)",
        width=800,
        height=600,
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        background_fill_color=Theme.Colors.BACKGROUND,
        border_fill_color=Theme.Colors.BORDER,
        sizing_mode="stretch_both",
    )

    plot.axis.axis_label_text_font_style = "bold"
    plot.grid.grid_line_alpha = 0.3
    plot.title.text_font_size = "14pt"  # type: ignore

    plot.line(
        data.reference.x,
        data.reference.y,
        line_width=Theme.Sizes.LINE_WIDTH,
        line_color=Theme.Colors.REFERENCE,
        line_dash="dashed",
        legend_label="Reference",
        alpha=0.8,
    )

    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    plot.legend.background_fill_alpha = 0.8

    return plot


def create_progress_plot(data: SimulationData) -> Figure:
    plot = figure(
        title="Path Progress",
        x_axis_label="Time (s)",
        y_axis_label="Path Parameter (m)",
        width=400,
        height=250,
        tools="pan,wheel_zoom,reset",
        background_fill_color=Theme.Colors.BACKGROUND,
        toolbar_location=None,
        sizing_mode="stretch_width",
    )

    times = data.time_steps * data.time_step

    plot.line(
        times,
        data.path_parameters,
        line_width=2,
        line_color=Theme.Colors.PRIMARY,
        alpha=0.8,
    )

    plot.line(
        times,
        np.full_like(times, data.path_length, dtype=float),
        line_width=2,
        line_color=Theme.Colors.SECONDARY,
        line_dash="dashed",
        alpha=0.6,
    )

    plot.grid.grid_line_alpha = 0.3
    return plot


def create_error_plot(data: SimulationData) -> Figure:
    errors = get_errors(data)
    times = data.time_steps * data.time_step

    plot = figure(
        title=data.error_label,
        x_axis_label="Time (s)",
        y_axis_label="Error (m)",
        width=400,
        height=250,
        tools="pan,wheel_zoom,reset",
        background_fill_color=Theme.Colors.BACKGROUND,
        toolbar_location=None,
        sizing_mode="stretch_width",
    )

    plot.line(
        times,
        errors,
        line_width=2,
        line_color=Theme.Colors.ACCENT,
        alpha=0.8,
    )

    plot.line(
        times,
        np.ones_like(times) * data.max_error,
        line_width=1,
        line_color=Theme.Colors.ACCENT_DARK,
        line_dash="dotted",
        alpha=0.5,
    )

    plot.grid.grid_line_alpha = 0.3
    return plot


def create_time_slider(timestep_count: int) -> Slider:
    return Slider(
        start=0,
        end=timestep_count - 1,
        value=0,
        step=1,
        title="Timestep",
        width=600,
        bar_color=Theme.Colors.PRIMARY,
    )


def create_play_button(slider: Slider, timestep_count: int) -> Button:
    button = Button(label="▶ Play", button_type="success", width=80)

    callback = CustomJS(
        args={
            "button": button,
            "slider": slider,
            "max_timestep": timestep_count - 1,
        },
        code="""
            if (button.label.includes("Play")) {
                button.label = "⏸ Pause";
                button.button_type = "warning";
                
                const speed = window.animation_speed || 50;
                window.animation_interval = setInterval(() => {
                    if (slider.value >= max_timestep) {
                        slider.value = 0;
                    } else {
                        slider.value += 1;
                    }
                }, speed);
            } else {
                button.label = "▶ Play";
                button.button_type = "success";
                clearInterval(window.animation_interval);
            }
        """,
    )
    button.js_on_click(callback)
    return button


def create_speed_selector() -> Button:
    button = Button(label="1x", button_type="light", width=50)

    callback = CustomJS(
        args={"button": button},
        code="""
            const speeds = [
                {label: "0.5x", ms: 100},
                {label: "1x", ms: 50},
                {label: "2x", ms: 25},
                {label: "4x", ms: 12},
            ];
            
            const current = speeds.findIndex(s => s.label === button.label);
            const next = (current + 1) % speeds.length;
            
            button.label = speeds[next].label;
            window.animation_speed = speeds[next].ms;
            
            if (window.animation_interval) {
                clearInterval(window.animation_interval);
                const slider = window.animation_slider;
                const max_timestep = window.animation_max_timestep;
                window.animation_interval = setInterval(() => {
                    if (slider.value >= max_timestep) {
                        slider.value = 0;
                    } else {
                        slider.value += 1;
                    }
                }, speeds[next].ms);
            }
        """,
    )
    button.js_on_click(callback)
    return button


def create_reset_button(slider: Slider) -> Button:
    button = Button(label="↺", button_type="light", width=40)
    callback = CustomJS(args={"slider": slider}, code="slider.value = 0;")
    button.js_on_click(callback)
    return button


def add_vehicle_to_plot(
    plot: Figure, source: ColumnDataSource, data: SimulationData
) -> None:
    match data.vehicle_type:
        case "car":
            add_car_to_plot(plot, source, data)
        case "triangle":
            plot.scatter(  # type: ignore
                "x",
                "y",
                source=source,
                size=Theme.Sizes.VEHICLE_SIZE,
                color=Theme.Colors.ACCENT,
                marker="triangle",
                angle="heading",
                legend_label="Vehicle",
                line_color=Theme.Colors.ACCENT_DARK,
                line_width=2,
            )


def add_car_to_plot(
    plot: Figure, source: ColumnDataSource, data: SimulationData
) -> None:
    plot.rect(  # type: ignore
        "x",
        "y",
        width=data.wheelbase,  # Dimensions are the other way around.
        height=data.vehicle_width,
        angle="heading",
        source=source,
        color=Theme.Colors.ACCENT,
        line_color=Theme.Colors.ACCENT_DARK,
        line_width=2,
        legend_label="Vehicle",
    )


def add_ghost_vehicle_to_plot(plot: Figure, source: ColumnDataSource) -> None:
    plot.scatter(  # type: ignore
        "x",
        "y",
        source=source,
        size=Theme.Sizes.VEHICLE_SIZE - 3,
        color=Theme.Colors.SECONDARY,
        marker="circle",
        alpha=0.5,
        legend_label="Path Parameter Position",
    )


def add_obstacles_to_plot(
    plot: Figure, source: ColumnDataSource, data: SimulationData
) -> None:
    if data.obstacle_positions_x is None or data.obstacle_positions_y is None:
        return

    obstacle_color = "#7f8c8d"
    obstacle_border = "#5a6263"

    plot.rect(  # type: ignore
        "x",
        "y",
        source=source,
        width=data.wheelbase,
        height=data.vehicle_width,
        angle="heading",
        color=obstacle_color,
        line_color=obstacle_border,
        line_width=2,
        alpha=0.8,
        legend_label="Obstacle",
    )


def add_obstacle_forecasts_to_plot(
    plot: Figure, source: ColumnDataSource, data: SimulationData
) -> None:
    if data.obstacle_forecast_x is None or data.obstacle_forecast_y is None:
        return

    if data.obstacle_forecast_x.shape[1] == 0 or data.obstacle_forecast_x.shape[2] == 0:
        return

    forecast_color = "#9b59b6"

    plot.multi_line(  # type: ignore
        "xs",
        "ys",
        source=source,
        line_width=2,
        line_color=forecast_color,
        line_alpha=0.6,
        legend_label="Obstacle Forecast",
    )

    add_forecast_arrows_to_plot(plot, source, forecast_color)


def add_forecast_arrows_to_plot(
    plot: Figure, source: ColumnDataSource, color: str
) -> None:
    arrow_length = 1.5

    plot.scatter(  # type: ignore
        "arrow_x",
        "arrow_y",
        source=source,
        marker="triangle",
        size=8,
        angle="arrow_heading",
        color=color,
        alpha=0.7,
    )


def add_trajectory_trace_to_plot(plot: Figure, source: ColumnDataSource) -> None:
    plot.line(  # type: ignore
        "x",
        "y",
        source=source,
        line_width=Theme.Sizes.LINE_WIDTH,
        line_color=Theme.Colors.PRIMARY,
        legend_label="Actual Path",
        alpha=0.8,
    )


def add_progress_marker_to_plot(plot: Figure, source: ColumnDataSource) -> None:
    plot.scatter(  # type: ignore
        "timestep",
        "path_parameter",
        source=source,
        size=Theme.Sizes.MARKER_SIZE,
        color=Theme.Colors.ACCENT,
        marker="circle",
        line_color=Theme.Colors.ACCENT_DARK,
        line_width=2,
    )


def add_error_marker_to_plot(plot: Figure, source: ColumnDataSource) -> None:
    plot.scatter(  # type: ignore
        "timestep",
        "error",
        source=source,
        size=Theme.Sizes.MARKER_SIZE,
        color=Theme.Colors.ACCENT,
        marker="circle",
        line_color=Theme.Colors.ACCENT_DARK,
        line_width=2,
    )


def create_slider_callback(
    *, sources: dict[str, ColumnDataSource], data: SimulationData
) -> CustomJS:
    return CustomJS(
        args={
            "vehicle": sources["vehicle"],
            "trajectory": sources["trajectory"],
            "progress": sources["progress"],
            "ghost": sources["ghost"],
            "error": sources["error"],
            "obstacles": sources["obstacles"],
            "forecast": sources["forecast"],
            "sim": sources["simulation"],
            "dt": data.time_step,
        },
        code="""
            const t = cb_obj.value;
            const s = sim.data;
            
            vehicle.data = {
                x: [s.positions_x[t]],
                y: [s.positions_y[t]],
                heading: [s.headings[t]]
            };
            
            trajectory.data = {
                x: s.positions_x.slice(0, t + 1),
                y: s.positions_y.slice(0, t + 1)
            };
            
            progress.data = {
                timestep: [t * dt],
                path_parameter: [s.path_parameters[t]]
            };
            
            ghost.data = {
                x: [s.ghost_x[t]],
                y: [s.ghost_y[t]]
            };
            
            error.data = {
                timestep: [t * dt],
                error: [s.errors[t]]
            };
            
            const obstacle_positions_x = s.obstacle_positions_x;
            const obstacle_positions_y = s.obstacle_positions_y;
            const obstacle_headings = s.obstacle_headings;

            obstacles.data = {
                x: obstacle_positions_x[t],
                y: obstacle_positions_y[t],
                heading: obstacle_headings[t]
            };
            
            // Update forecast lines
            const forecast_x = s.obstacle_forecast_x;
            const forecast_y = s.obstacle_forecast_y;
            const forecast_headings = s.obstacle_forecast_headings;
            
            if (forecast_x && forecast_x.length > 0 && forecast_x[0].length > 0) {
                const obstacle_count = forecast_x[t][0].length;
                const xs = [];
                const ys = [];
                const headings = [];
                const arrow_x = [];
                const arrow_y = [];
                const arrow_heading = [];
                
                for (let k = 0; k < obstacle_count; k++) {
                    const horizon = forecast_x[t].length;
                    const x_line = [];
                    const y_line = [];
                    const h_line = [];
                    
                    for (let h = 0; h < horizon; h++) {
                        x_line.push(forecast_x[t][h][k]);
                        y_line.push(forecast_y[t][h][k]);
                        h_line.push(forecast_headings[t][h][k]);
                    }
                    
                    xs.push(x_line);
                    ys.push(y_line);
                    headings.push(h_line);
                    
                    // Add arrow at the end of each forecast line
                    const last_idx = horizon - 1;
                    arrow_x.push(x_line[last_idx]);
                    arrow_y.push(y_line[last_idx]);
                    arrow_heading.push(h_line[last_idx]);
                }
                
                forecast.data = {
                    xs: xs,
                    ys: ys,
                    headings: headings,
                    arrow_x: arrow_x,
                    arrow_y: arrow_y,
                    arrow_heading: arrow_heading,
                };
            }
        """,
    )


def info_section_placeholder() -> Div:
    return Div(text="Simulation info will appear here...", width=400, height=180)


def setup_info_updates(
    info_div: Div, slider: Slider, data: SimulationData, target: Figure
) -> None:
    args = {
        "info_div": info_div,
        "slider": slider,
        "positions_x": data.positions_x.tolist(),
        "positions_y": data.positions_y.tolist(),
        "headings": data.headings.tolist(),
        "path_parameters": data.path_parameters.tolist(),
        "path_length": data.path_length,
        "dt": data.time_step,
        "template": INFO_TEMPLATE,
    }

    slider.js_on_change("value", callback := CustomJS(args=args, code=INFO_UPDATE_CODE))
    target.js_on_event(DocumentReady, callback)


def create_layout(
    *,
    trajectory_plot: Figure,
    progress_plot: Figure,
    error_plot: Figure,
    info_section: Div,
    time_slider: Slider,
    play_button: Button,
    speed_selector: Button,
    reset_button: Button,
) -> Column:
    title = Div(
        text="""
        <h1 style="margin: 0 0 5px 0; color: #2c3e50; font-family: system-ui;">
            MPCC Simulation
        </h1>
        """,
        sizing_mode="stretch_width",
    )

    controls = row(
        play_button,
        reset_button,
        speed_selector,
        Div(width=20),
        time_slider,
        styles={
            "align-items": "center",
            "padding": "10px 15px",
            "background": "#ecf0f1",
            "border-radius": "8px",
            "margin-top": "15px",
        },
        sizing_mode="stretch_width",
    )

    time_slider.sizing_mode = "stretch_width"

    right_panel = column(
        info_section,
        Div(height=10),
        progress_plot,
        Div(height=10),
        error_plot,
        width=400,
    )

    info_section.sizing_mode = "stretch_width"

    main_content = row(
        trajectory_plot,
        Div(width=20),
        right_panel,
        sizing_mode="stretch_both",
    )

    return column(
        title,
        main_content,
        controls,
        styles={"padding": "20px", "font-family": "system-ui, sans-serif"},
        sizing_mode="stretch_both",
    )


async def save(layout: Column, *, to: AsyncPath) -> None:
    async with to.open("w", encoding="utf-8") as f:
        await f.write(file_html(layout, CDN, title="Simulation Visualization"))
