import { z } from "zod";

export namespace Types {
    export const Vehicle = z.enum(["triangle", "car"]);
    export const Scale = z.enum(["linear", "log"]);
}

export namespace Plot {
    export const Series = z.object({
        label: z.string(),
        values: z.array(z.number()),
        color: z.string().optional(),
    });

    export const Bound = z.object({
        values: z.union([z.array(z.number()), z.number()]),
        label: z.string().optional(),
    });

    export const Band = z.object({
        lower: z.array(z.number()),
        upper: z.array(z.number()),
        color: z.string().optional(),
        label: z.string().optional(),
    });

    export const Additional = z.object({
        id: z.string(),
        name: z.string(),
        series: z.array(Series),
        yAxisLabel: z.string(),
        upperBound: Bound.optional(),
        lowerBound: Bound.optional(),
        bands: z.array(Band).optional(),
        yAxisScale: Types.Scale.optional(),
        group: z.string().optional(),
    });
}

export namespace Road {
    export const MarkingType = z.enum(["solid", "dashed", "none"]);

    export const Lane = z.object({
        x: z.array(z.number()),
        y: z.array(z.number()),
        boundaries: z.tuple([z.number(), z.number()]),
        markings: z.tuple([MarkingType, MarkingType]),
    });

    export const Network = z.object({
        lanes: z.array(Lane),
    });
}

export namespace Visualizable {
    export const ReferenceTrajectory = z.object({
        x: z.array(z.number()),
        y: z.array(z.number()),
    });

    export const SimulationInfo = z.object({
        pathLength: z.number(),
        timeStep: z.number(),
        wheelbase: z.number().optional().default(2.5),
        vehicleWidth: z.number().optional().default(1.2),
        vehicleType: Types.Vehicle.optional().default("triangle"),
    });

    export const EgoGhost = z.object({
        x: z.array(z.number()),
        y: z.array(z.number()),
    });

    export const Ego = z.object({
        x: z.array(z.number()),
        y: z.array(z.number()),
        heading: z.array(z.number()),
        pathParameter: z.array(z.number()),
        ghost: EgoGhost.optional(),
    });

    export const PlannedTrajectory = z.object({
        x: z.array(z.array(z.number())),
        y: z.array(z.array(z.number())),
    });

    export const PlannedTrajectories = z.object({
        optimal: PlannedTrajectory.optional(),
        nominal: PlannedTrajectory.optional(),
    });

    export const ObstacleForecast = z.object({
        x: z.array(z.array(z.array(z.number().nullable()))),
        y: z.array(z.array(z.array(z.number().nullable()))),
        heading: z.array(z.array(z.array(z.number().nullable()))),
        covariance: z.array(z.array(z.array(z.array(z.array(z.number().nullable()))))).optional(),
    });

    export const Obstacles = z.object({
        x: z.array(z.array(z.number().nullable())),
        y: z.array(z.array(z.number().nullable())),
        heading: z.array(z.array(z.number().nullable())),
        forecast: ObstacleForecast.optional(),
    });

    export const SimulationResult = z.object({
        info: SimulationInfo,
        reference: ReferenceTrajectory,
        ego: Ego,
        trajectories: PlannedTrajectories.optional(),
        obstacles: Obstacles.optional(),
        network: Road.Network.optional(),
        additionalPlots: z.array(Plot.Additional).optional(),
    });
}

export type SimulationDataInput = z.input<typeof Visualizable.SimulationResult>;
export type SimulationDataOutput = z.output<typeof Visualizable.SimulationResult>;

export function parseSimulationData(data: unknown): SimulationDataOutput {
    return Visualizable.SimulationResult.parse(data);
}
