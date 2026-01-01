export type VehicleType = "triangle" | "car";

export interface ReferenceTrajectory {
    x: number[];
    y: number[];
}

export interface SimulationData {
    reference: ReferenceTrajectory;
    positions_x: number[];
    positions_y: number[];
    headings: number[];
    path_parameters: number[];
    path_length: number;
    time_step: number;
    errors?: number[];
    ghost_x?: number[];
    ghost_y?: number[];
    max_error: number;
    error_label: string;
    vehicle_type: VehicleType;
    wheelbase: number;
    vehicle_width: number;
    obstacle_positions_x?: number[][];
    obstacle_positions_y?: number[][];
    obstacle_headings?: number[][];
    obstacle_forecast_x?: number[][][];
    obstacle_forecast_y?: number[][][];
    obstacle_forecast_heading?: number[][][];
    obstacle_forecast_covariance?: number[][][][][];
}

export interface ProcessedSimulationData extends SimulationData {
    timestep_count: number;
}

export interface EllipseParameters {
    width: number;
    height: number;
    angle: number;
}

export interface FlatEllipseData {
    x: number[];
    y: number[];
    width: number[];
    height: number[];
    angle: number[];
}

export interface ForecastData {
    xs: number[][];
    ys: number[][];
    headings: number[][];
    arrow_x: number[];
    arrow_y: number[];
    arrow_heading: number[];
}

export interface CliOptions {
    output: string;
    title: string;
    width: number;
    height: number;
}

export interface BatchOptions {
    pattern: string;
    recursive: boolean;
}

export interface WatchOptions {
    port: number;
}

export interface PlotDimensions {
    width: number;
    height: number;
    margin: { top: number; right: number; bottom: number; left: number };
}
