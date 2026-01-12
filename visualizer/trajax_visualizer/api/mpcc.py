from typing import Sequence
from dataclasses import dataclass

from trajax import (
    types,
    Trajectory,
    PathParameters,
    ReferencePoints,
    ObstacleStates,
    ControlInputSequence,
    Weights,
    Control,
    Risk,
)

from trajax_visualizer.api.simulation import (
    SimulationVisualizer,
    SimulationData,
    SimulationInfo,
    ReferenceTrajectory,
    Ego,
    EgoGhost,
    PlannedTrajectories,
    PlannedTrajectory,
    Obstacles,
    ObstacleForecast,
    PlotSeries,
    PlotBound,
    PlotBand,
    AdditionalPlot,
    ObstacleForecastCovarianceArray,
)

from numtypes import Array, Dim1, Dims, D

import numpy as np

type PhysicalStateSequence = (
    types.numpy.bicycle.StateSequence | types.jax.bicycle.StateSequence
)
type VirtualStateSequence = (
    types.numpy.simple.StateSequence | types.jax.simple.StateSequence
)
type AugmentedStateSequence = types.augmented.State[
    PhysicalStateSequence, VirtualStateSequence
]


@dataclass(kw_only=True, frozen=True)
class MpccSimulationResult:
    reference: Trajectory[PathParameters, ReferencePoints]
    states: AugmentedStateSequence
    contouring_errors: Array[Dim1]
    lag_errors: Array[Dim1]
    time_step_size: float
    wheelbase: float
    max_contouring_error: float
    max_lag_error: float
    optimal_trajectories: Sequence[AugmentedStateSequence] | None = None
    nominal_trajectories: Sequence[AugmentedStateSequence] | None = None
    obstacles: ObstacleStates | None = None
    obstacle_forecasts: Sequence[ObstacleStates] | None = None
    controls: Sequence[Control[ControlInputSequence, Weights]] | None = None
    risks: Sequence[Risk] | None = None


@dataclass(frozen=True)
class MpccVisualizer:
    """Visualizer for MPCC simulation results."""

    inner: SimulationVisualizer
    reference_sample_count: int

    @staticmethod
    def create(
        *,
        output: str = "mpcc-simulation",
        reference_sample_count: int = 200,
        output_directory: str | None = None,
    ) -> "MpccVisualizer":
        return MpccVisualizer(
            SimulationVisualizer.create(
                output=output, output_directory=output_directory
            ),
            reference_sample_count,
        )

    async def __call__(self, data: MpccSimulationResult, *, key: str) -> None:
        await self.inner(self.extract(data), key=key)

    async def can_visualize(self, data: object) -> bool:
        return isinstance(data, MpccSimulationResult)

    def extract(self, result: MpccSimulationResult) -> SimulationData:
        return SimulationData.create(
            info=self.info_from(result),
            reference=self.reference_trajectory_from(result),
            ego=self.ego_from(result),
            trajectories=self.planned_trajectories_from(result),
            obstacles=self.obstacles_from(result),
            additional_plots=self.additional_plots_from(result),
        )

    def info_from(self, result: MpccSimulationResult) -> SimulationInfo:
        return SimulationInfo(
            path_length=result.reference.path_length,
            time_step=result.time_step_size,
            wheelbase=result.wheelbase,
            vehicle_type="car",
        )

    def reference_trajectory_from(
        self, result: MpccSimulationResult
    ) -> ReferenceTrajectory:
        path_parameters = np.linspace(
            0, result.reference.path_length, self.reference_sample_count
        ).reshape(-1, 1)

        reference_points = result.reference.query(
            types.numpy.path_parameters(path_parameters)
        )

        return ReferenceTrajectory(
            x=reference_points.x()[:, 0], y=reference_points.y()[:, 0]
        )

    def ego_from(self, result: MpccSimulationResult) -> Ego:
        return Ego(
            x=result.states.physical.x(),
            y=result.states.physical.y(),
            heading=result.states.physical.heading(),
            path_parameter=(path_parameters := self.path_parameters_from(result)),
            ghost=self.ego_ghost_from(result, path_parameters),
        )

    def path_parameters_from(self, result: MpccSimulationResult) -> Array[Dim1]:
        # NOTE: We assume the first virtual dimension is the path parameter.
        return np.asarray(result.states.virtual)[:, 0]

    def ego_ghost_from(
        self, result: MpccSimulationResult, path_parameters: Array[Dim1]
    ) -> EgoGhost:
        positions = result.reference.query(
            types.numpy.path_parameters(path_parameters.reshape(-1, 1))
        )

        return EgoGhost(x=positions.x()[:, 0], y=positions.y()[:, 0])

    def planned_trajectories_from(
        self, result: MpccSimulationResult
    ) -> PlannedTrajectories | None:
        if result.optimal_trajectories is None:
            return

        return PlannedTrajectories(
            optimal=self.planned_trajectory_from(result.optimal_trajectories),
            nominal=self.planned_trajectory_from(result.nominal_trajectories),
        )

    def planned_trajectory_from(
        self, trajectories: Sequence[AugmentedStateSequence] | None
    ) -> PlannedTrajectory | None:
        if trajectories is None:
            return

        return PlannedTrajectory(
            x=np.stack([it.physical.x() for it in trajectories], axis=0),
            y=np.stack([it.physical.y() for it in trajectories], axis=0),
        )

    def obstacles_from(self, result: MpccSimulationResult) -> Obstacles | None:
        if result.obstacles is None:
            return

        return Obstacles(
            x=result.obstacles.x(),
            y=result.obstacles.y(),
            heading=result.obstacles.heading(),
            forecast=self.obstacle_forecast_from(result),
        )

    def obstacle_forecast_from(
        self, result: MpccSimulationResult
    ) -> ObstacleForecast | None:
        if result.obstacle_forecasts is None:
            return

        return ObstacleForecast(
            x=np.stack([it.x() for it in result.obstacle_forecasts], axis=0),
            y=np.stack([it.y() for it in result.obstacle_forecasts], axis=0),
            heading=np.stack(
                [it.heading() for it in result.obstacle_forecasts], axis=0
            ),
            covariance=self.obstacle_forecast_covariance_from(result),
        )

    def obstacle_forecast_covariance_from(
        self, result: MpccSimulationResult
    ) -> ObstacleForecastCovarianceArray | None:
        if (forecasts := result.obstacle_forecasts) is None or (
            template_covariance := self.template_covariance_from(forecasts)
        ) is None:
            return

        no_covariance = np.zeros_like(template_covariance)

        return np.stack(
            [
                self.position_covariance_from(covariance)
                if (covariance := it.covariance()) is not None
                else no_covariance
                for it in forecasts
            ],
            axis=0,
        )

    def template_covariance_from(
        self, forecasts: Sequence[ObstacleStates]
    ) -> Array | None:
        for it in forecasts:
            if (covariance := it.covariance()) is not None:
                return self.position_covariance_from(covariance)

    def position_covariance_from[T: int = int, D_x: int = int, K: int = int](
        self, covariance: Array[Dims[T, D_x, D_x, K]]
    ) -> Array[Dims[T, D[2], D[2], K]]:
        # NOTE: We assume first two dimensions correspond to covariance of (x, y).
        return covariance[:, :2, :2, :]

    def additional_plots_from(
        self, result: MpccSimulationResult
    ) -> list[AdditionalPlot]:
        path_parameters = self.path_parameters_from(result)
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
                        label="Contouring Error",
                        values=np.asarray(result.contouring_errors).flatten(),
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
                        label="Lag Error",
                        values=np.asarray(result.lag_errors).flatten(),
                        color="#9b59b6",
                    )
                ],
                y_axis_label="Error (m)",
                upper_bound=PlotBound(values=result.max_lag_error),
                lower_bound=PlotBound(values=-result.max_lag_error),
                group="errors",
            ),
        ]

        if risk_plot := self.risk_plot_from(result):
            plots.append(risk_plot)

        return plots

    def risk_plot_from(self, result: MpccSimulationResult) -> AdditionalPlot | None:
        if (
            result.risks is None
            or result.controls is None
            or len(result.risks) == 0
            or len(result.controls) == 0
        ):
            return

        risks = np.stack([np.asarray(risk) for risk in result.risks], axis=0).sum(
            axis=1
        )
        weights = np.stack(
            [np.asarray(c.debug.trajectory_weights) for c in result.controls], axis=0
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
