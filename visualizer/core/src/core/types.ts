export namespace Types {
    export type Vehicle = "triangle" | "car";
    export type Scale = "linear" | "log";
}

export namespace Plot {
    export interface Series {
        label: string;
        values: number[];
        color?: string;
    }

    export interface Bound {
        values: number[] | number;
        label?: string;
    }

    export interface Band {
        lower: number[];
        upper: number[];
        color?: string;
        label?: string;
    }

    export interface Additional {
        id: string;
        name: string;
        series: Series[];
        yAxisLabel: string;
        upperBound?: Bound;
        lowerBound?: Bound;
        bands?: Band[];
        yAxisScale?: Types.Scale;
        group?: string;
    }
}

export namespace Visualizable {
    export interface ReferenceTrajectory {
        x: number[];
        y: number[];
    }

    export interface SimulationInfo {
        pathLength: number;
        timeStep: number;
        wheelbase: number;
        vehicleWidth: number;
        vehicleType: Types.Vehicle;
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

    export interface SimulationResult {
        info: SimulationInfo;
        reference: ReferenceTrajectory;
        ego: Ego;
        trajectories?: PlannedTrajectories;
        obstacles?: Obstacles;
        additionalPlots?: Plot.Additional[];
    }

    export interface ProcessedSimulationResult extends SimulationResult {
        timestepCount: number;
    }
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
