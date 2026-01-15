export namespace Types {
    export type Vehicle = "triangle" | "car";
    export type Scale = "linear" | "log";
    export type MarkingType = "solid" | "dashed" | "none";
}

export namespace Arrays {
    /** Shape: [T, K] */
    export type ObstacleCoordinates = number[][];

    /** Shape: [T, H, K] */
    export type ForecastCoordinates = (number | null)[][][];

    /** Shape: [T, H, 2, 2, K] */
    export type ForecastCovariances = (number | null)[][][][][];

    /** Shape: [T, H] */
    export type PlannedCoordinates = number[][];
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

export namespace Road {
    export interface Lane {
        x: number[];
        y: number[];
        boundaries: [number, number];
        markings: [Types.MarkingType, Types.MarkingType];
    }

    export interface Network {
        lanes: Lane[];
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
        x: Arrays.PlannedCoordinates;
        y: Arrays.PlannedCoordinates;
    }

    export interface PlannedTrajectories {
        optimal?: PlannedTrajectory;
        nominal?: PlannedTrajectory;
    }

    export interface ObstacleForecast {
        x: Arrays.ForecastCoordinates;
        y: Arrays.ForecastCoordinates;
        heading: Arrays.ForecastCoordinates;
        covariance?: Arrays.ForecastCovariances;
    }

    export interface Obstacles {
        x: Arrays.ObstacleCoordinates;
        y: Arrays.ObstacleCoordinates;
        heading: Arrays.ObstacleCoordinates;
        forecast?: ObstacleForecast;
    }

    export interface SimulationResult {
        info: SimulationInfo;
        reference: ReferenceTrajectory;
        ego: Ego;
        trajectories?: PlannedTrajectories;
        obstacles?: Obstacles;
        network?: Road.Network;
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
