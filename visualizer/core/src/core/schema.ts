import { z } from "zod";

const ReferenceTrajectorySchema = z.object({
    x: z.array(z.number()),
    y: z.array(z.number()),
});

const PlotSeriesSchema = z.object({
    label: z.string(),
    values: z.array(z.number()),
    color: z.string().optional(),
});

const PlotBoundSchema = z.object({
    values: z.union([z.array(z.number()), z.number()]),
    label: z.string().optional(),
});

const PlotBandSchema = z.object({
    lower: z.array(z.number()),
    upper: z.array(z.number()),
    color: z.string().optional(),
    label: z.string().optional(),
});

const AdditionalPlotSchema = z.object({
    id: z.string(),
    name: z.string(),
    series: z.array(PlotSeriesSchema),
    upperBound: PlotBoundSchema.optional(),
    lowerBound: PlotBoundSchema.optional(),
    bands: z.array(PlotBandSchema).optional(),
    yAxisLabel: z.string(),
    yAxisScale: z.enum(["linear", "log"]).optional(),
    group: z.string().optional(),
});

const SimulationInfoSchema = z.object({
    pathLength: z.number(),
    timeStep: z.number().optional().default(0.1),
    wheelbase: z.number().optional().default(2.5),
    vehicleWidth: z.number().optional().default(1.2),
    vehicleType: z.enum(["triangle", "car"]).optional().default("triangle"),
});

const EgoGhostSchema = z.object({
    x: z.array(z.number()),
    y: z.array(z.number()),
});

const EgoSchema = z.object({
    x: z.array(z.number()),
    y: z.array(z.number()),
    heading: z.array(z.number()),
    pathParameter: z.array(z.number()),
    ghost: EgoGhostSchema.optional(),
});

const PlannedTrajectorySchema = z.object({
    x: z.array(z.array(z.number())),
    y: z.array(z.array(z.number())),
});

const PlannedTrajectoriesSchema = z.object({
    optimal: PlannedTrajectorySchema.optional(),
    nominal: PlannedTrajectorySchema.optional(),
});

const ObstacleForecastSchema = z.object({
    x: z.array(z.array(z.array(z.number().nullable()))),
    y: z.array(z.array(z.array(z.number().nullable()))),
    heading: z.array(z.array(z.array(z.number().nullable()))),
    covariance: z.array(z.array(z.array(z.array(z.array(z.number().nullable()))))).optional(),
});

const ObstaclesSchema = z.object({
    x: z.array(z.array(z.number())),
    y: z.array(z.array(z.number())),
    heading: z.array(z.array(z.number())),
    forecast: ObstacleForecastSchema.optional(),
});

export const SimulationDataSchema = z.object({
    info: SimulationInfoSchema,
    reference: ReferenceTrajectorySchema,
    ego: EgoSchema,
    trajectories: PlannedTrajectoriesSchema.optional(),
    obstacles: ObstaclesSchema.optional(),
    additionalPlots: z.array(AdditionalPlotSchema).optional(),
});

export type SimulationDataInput = z.input<typeof SimulationDataSchema>;
export type SimulationDataOutput = z.output<typeof SimulationDataSchema>;

export function parseSimulationData(data: unknown): SimulationDataOutput {
    return SimulationDataSchema.parse(data);
}
