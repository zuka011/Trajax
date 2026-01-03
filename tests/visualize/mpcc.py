from typing import Sequence
from dataclasses import dataclass

from trajax import (
    types,
    Trajectory,
    ObstacleStates,
    StateSequence,
    ControlInputSequence,
    Weights,
    Control,
    Risk,
)

import numpy as np
from numtypes import Array, Dim1, Dim2

from tests.visualize.simulation import (
    SimulationVisualizer,
    SimulationData,
    ReferenceTrajectory,
    ObstacleForecast,
    ObstacleForecastCovariance,
    AdditionalPlot,
    PlotSeries,
    PlotBound,
    PlotBand,
)


type PhysicalState = types.numpy.bicycle.State | types.jax.bicycle.State
type VirtualState = types.numpy.simple.State | types.jax.simple.State
type AugmentedState = types.augmented.State[PhysicalState, VirtualState]


@dataclass(kw_only=True, frozen=True)
class MpccSimulationResult:
    reference: Trajectory
    states: Sequence[AugmentedState]
    optimal_trajectories: Sequence[StateSequence]
    nominal_trajectories: Sequence[StateSequence]
    contouring_errors: Array[Dim1]
    lag_errors: Array[Dim1]
    wheelbase: float
    max_contouring_error: float
    max_lag_error: float
    obstacles: Sequence[ObstacleStates] = ()
    controls: Sequence[Control[ControlInputSequence, Weights]] = ()
    risks: Sequence[Risk] = ()


@dataclass(frozen=True)
class MpccVisualizer:
    """Visualizer for MPCC simulation results."""

    inner: SimulationVisualizer
    reference_sample_count: int

    @staticmethod
    def create(
        *, output: str = "mpcc-simulation", reference_sample_count: int = 200
    ) -> "MpccVisualizer":
        return MpccVisualizer(
            SimulationVisualizer.create(output=output), reference_sample_count
        )

    async def __call__(self, data: MpccSimulationResult, *, key: str) -> None:
        await self.inner(self.extract(data), key=key)

    async def can_visualize(self, data: object) -> bool:
        return isinstance(data, MpccSimulationResult)

    def extract(self, result: MpccSimulationResult) -> SimulationData:
        reference = self.sample_reference_trajectory(
            result.reference, result.reference.path_length
        )
        path_parameters = np.array([state.virtual.array[0] for state in result.states])
        ghost_positions = self.query_ghost_positions(result.reference, path_parameters)
        obstacle_x, obstacle_y, obstacle_heading = self.obstacle_positions_from(
            result.obstacles
        )
        forecast_x, forecast_y, forecast_heading = self.obstacle_forecasts_from(
            result.obstacles
        )
        forecast_covariance = self.obstacle_forecast_covariance_from(result.obstacles)
        additional_plots = self.build_additional_plots(result, path_parameters)

        return SimulationData.create(
            reference=reference,
            positions_x=np.array([state.physical.x for state in result.states]),
            positions_y=np.array([state.physical.y for state in result.states]),
            headings=np.array([state.physical.theta for state in result.states]),
            path_parameters=path_parameters,
            path_length=result.reference.path_length,
            ghost_x=ghost_positions.x,
            ghost_y=ghost_positions.y,
            wheelbase=result.wheelbase,
            vehicle_type="car",
            obstacle_positions_x=obstacle_x,
            obstacle_positions_y=obstacle_y,
            obstacle_headings=obstacle_heading,
            obstacle_forecast_x=forecast_x,
            obstacle_forecast_y=forecast_y,
            obstacle_forecast_heading=forecast_heading,
            obstacle_forecast_covariance=forecast_covariance,
            additional_plots=additional_plots,
        )

    def build_additional_plots(
        self, result: MpccSimulationResult, path_parameters: Array[Dim1]
    ) -> list[AdditionalPlot]:
        plots = [
            AdditionalPlot(
                id="progress",
                name="Path Progress",
                series=[PlotSeries(label="Progress", values=path_parameters)],
                y_axis_label="Progress (m)",
                upper_bound=PlotBound(
                    values=result.reference.path_length, label="Path Length"
                ),
            ),
            AdditionalPlot(
                id="contouring-error",
                name="Contouring Error",
                series=[
                    PlotSeries(
                        label="Contouring Error", values=result.contouring_errors
                    )
                ],
                y_axis_label="Error (m)",
                upper_bound=PlotBound(values=result.max_contouring_error),
                lower_bound=PlotBound(values=-result.max_contouring_error),
                group="errors",
            ),
            AdditionalPlot(
                id="lag-error",
                name="Lag Error",
                series=[
                    PlotSeries(
                        label="Lag Error", values=result.lag_errors, color="#9b59b6"
                    )
                ],
                y_axis_label="Error (m)",
                upper_bound=PlotBound(values=result.max_lag_error),
                lower_bound=PlotBound(values=-result.max_lag_error),
                group="errors",
            ),
        ]

        if risk_plot := self.build_risk_plot(result):
            plots.append(risk_plot)

        return plots

    def build_risk_plot(self, result: MpccSimulationResult) -> AdditionalPlot | None:
        if len(result.risks) == 0 or len(result.controls) == 0:
            return None

        risks = np.array([np.asarray(risk) for risk in result.risks]).sum(axis=1)
        weights = np.array(
            [np.asarray(c.debug.trajectory_weights) for c in result.controls]
        )

        # NOTE: avoid zero risks for log scale plotting
        risks = np.maximum(risks, 1e-10)
        selected = (risks * weights).sum(axis=1)

        return AdditionalPlot(
            id="risk",
            name="Risk",
            series=[
                PlotSeries(label="Selected", values=selected, color="#e63946"),
                PlotSeries(
                    label="Median", values=np.median(risks, axis=1), color="#457b9d"
                ),
            ],
            bands=[
                PlotBand(
                    lower=np.percentile(risks, 10, axis=1),
                    upper=np.percentile(risks, 90, axis=1),
                    color="#adb5bd",
                    label="10-90%",
                ),
                PlotBand(
                    lower=np.percentile(risks, 25, axis=1),
                    upper=np.percentile(risks, 75, axis=1),
                    color="#457b9d",
                    label="25-75%",
                ),
            ],
            y_axis_label="Risk",
            y_axis_scale="log",
        )

    def query_ghost_positions(
        self, trajectory: Trajectory, path_parameters: Array[Dim1]
    ) -> ReferenceTrajectory:
        query_params = types.numpy.path_parameters(path_parameters.reshape(-1, 1))
        points = trajectory.query(query_params)
        return ReferenceTrajectory(x=points.x()[:, 0], y=points.y()[:, 0])

    def sample_reference_trajectory(
        self, trajectory: Trajectory, path_length: float
    ) -> ReferenceTrajectory:
        path_parameters = np.linspace(
            0, path_length, self.reference_sample_count
        ).reshape(-1, 1)

        query_params = types.numpy.path_parameters(path_parameters)
        reference_points = trajectory.query(query_params)

        return ReferenceTrajectory(
            x=reference_points.x()[:, 0], y=reference_points.y()[:, 0]
        )

    def obstacle_positions_from(
        self, obstacles: Sequence[ObstacleStates]
    ) -> tuple[Array[Dim2] | None, Array[Dim2] | None, Array[Dim2] | None]:
        if len(obstacles) == 0:
            return None, None, None

        return (
            np.array([it.x()[0] for it in obstacles]),
            np.array([it.y()[0] for it in obstacles]),
            np.array([it.heading()[0] for it in obstacles]),
        )

    def obstacle_forecasts_from(
        self, obstacles: Sequence[ObstacleStates]
    ) -> tuple[
        ObstacleForecast | None, ObstacleForecast | None, ObstacleForecast | None
    ]:
        if len(obstacles) == 0:
            return None, None, None

        return (
            np.array([it.x() for it in obstacles]),
            np.array([it.y() for it in obstacles]),
            np.array([it.heading() for it in obstacles]),
        )

    def obstacle_forecast_covariance_from(
        self, obstacles: Sequence[ObstacleStates]
    ) -> ObstacleForecastCovariance | None:
        if len(obstacles) == 0:
            return None

        covariances = [it.covariance() for it in obstacles]

        if all(cov is None for cov in covariances):
            return None

        template_covariance = next(cov for cov in covariances if cov is not None)

        # NOTE: we assume the first two dimensions are x and y
        return np.array(
            [
                cov[:, :2, :2, :]
                if cov is not None
                else np.zeros_like(template_covariance[:, :2, :2, :])
                for cov in covariances
            ]
        )
