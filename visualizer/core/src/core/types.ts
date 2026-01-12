export type VehicleType = "triangle" | "car";
export type ScaleType = "linear" | "log";

export interface ReferenceTrajectory {
    x: number[];
    y: number[];
}

export interface PlotSeries {
    label: string;
    values: number[];
    color?: string;
}

export interface PlotBound {
    values: number[] | number;
    label?: string;
}

export interface PlotBand {
    lower: number[];
    upper: number[];
    color?: string;
    label?: string;
}

export interface AdditionalPlot {
    id: string;
    name: string;
    series: PlotSeries[];
    upperBound?: PlotBound;
    lowerBound?: PlotBound;
    bands?: PlotBand[];
    yAxisScale?: ScaleType;
    yAxisLabel: string;
    group?: string;
}

export interface SimulationInfo {
    pathLength: number;
    timeStep: number;
    wheelbase: number;
    vehicleWidth: number;
    vehicleType: VehicleType;
}

export interface EgoGhost {
    x: number[];
    y: number[];
}

export interface Ego {
    x: number[];
    y: number[];
    heading: number[];
    pathParameter: number[];
    ghost?: EgoGhost;
}

export interface PlannedTrajectory {
    x: number[][];
    y: number[][];
}

export interface PlannedTrajectories {
    optimal?: PlannedTrajectory;
    nominal?: PlannedTrajectory;
}

export interface ObstacleForecast {
    x: (number | null)[][][];
    y: (number | null)[][][];
    heading: (number | null)[][][];
    covariance?: (number | null)[][][][][];
}

export interface Obstacles {
    x: number[][];
    y: number[][];
    heading: number[][];
    forecast?: ObstacleForecast;
}

export interface SimulationData {
    info: SimulationInfo;
    reference: ReferenceTrajectory;
    ego: Ego;
    trajectories?: PlannedTrajectories;
    obstacles?: Obstacles;
    additionalPlots?: AdditionalPlot[];
}

export interface ProcessedSimulationData extends SimulationData {
    timestepCount: number;
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
    arrowX: number[];
    arrowY: number[];
    arrowHeading: number[];
}
