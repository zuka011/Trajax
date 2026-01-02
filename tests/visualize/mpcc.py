from typing import Sequence
from dataclasses import dataclass

from trajax import types, Trajectory, ObstacleStates

import numpy as np
from numtypes import Array, Dim1, Dim2

from tests.visualize.simulation import (
    SimulationVisualizer,
    SimulationData,
    ReferenceTrajectory,
    ObstacleForecast,
    ObstacleForecastCovariance,
)


type PhysicalState = types.numpy.bicycle.State | types.jax.bicycle.State
type VirtualState = types.numpy.simple.State | types.jax.simple.State
type AugmentedState = types.augmented.State[PhysicalState, VirtualState]


@dataclass(kw_only=True, frozen=True)
class MpccSimulationResult:
    reference: Trajectory
    states: Sequence[AugmentedState]
    contouring_errors: Array[Dim1]
    lag_errors: Array[Dim1]
    wheelbase: float
    max_contouring_error: float
    max_lag_error: float
    obstacles: Sequence[ObstacleStates] = ()


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

        return SimulationData(
            reference=reference,
            positions_x=np.array([state.physical.x for state in result.states]),
            positions_y=np.array([state.physical.y for state in result.states]),
            headings=np.array([state.physical.theta for state in result.states]),
            path_parameters=path_parameters,
            path_length=result.reference.path_length,
            ghost_x=ghost_positions.x,
            ghost_y=ghost_positions.y,
            errors=result.contouring_errors,
            wheelbase=result.wheelbase,
            max_error=result.max_contouring_error,
            error_label="Contouring Error",
            vehicle_type="car",
            obstacle_positions_x=obstacle_x,
            obstacle_positions_y=obstacle_y,
            obstacle_headings=obstacle_heading,
            obstacle_forecast_x=forecast_x,
            obstacle_forecast_y=forecast_y,
            obstacle_forecast_heading=forecast_heading,
            obstacle_forecast_covariance=forecast_covariance,
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
