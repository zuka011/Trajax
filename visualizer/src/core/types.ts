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

export interface SimulationData {
    reference: ReferenceTrajectory;
    positionsX: number[];
    positionsY: number[];
    headings: number[];
    pathParameters: number[];
    pathLength: number;
    timeStep: number;
    ghostX?: number[];
    ghostY?: number[];
    vehicleType: VehicleType;
    wheelbase: number;
    vehicleWidth: number;
    obstaclePositionsX?: number[][];
    obstaclePositionsY?: number[][];
    obstacleHeadings?: number[][];
    obstacleForecastX?: number[][][];
    obstacleForecastY?: number[][][];
    obstacleForecastHeading?: number[][][];
    obstacleForecastCovariance?: number[][][][][];
    optimalTrajectoryX?: number[][];
    optimalTrajectoryY?: number[][];
    nominalTrajectoryX?: number[][];
    nominalTrajectoryY?: number[][];
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
