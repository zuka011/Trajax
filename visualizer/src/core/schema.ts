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

export const SimulationDataSchema = z.object({
    reference: ReferenceTrajectorySchema,
    positionsX: z.array(z.number()),
    positionsY: z.array(z.number()),
    headings: z.array(z.number()),
    pathParameters: z.array(z.number()),
    pathLength: z.number(),
    timeStep: z.number().default(0.1),
    ghostX: z.array(z.number()).optional(),
    ghostY: z.array(z.number()).optional(),
    vehicleType: z.enum(["triangle", "car"]).default("triangle"),
    wheelbase: z.number().default(2.5),
    vehicleWidth: z.number().default(1.2),
    obstaclePositionsX: z.array(z.array(z.number())).optional(),
    obstaclePositionsY: z.array(z.array(z.number())).optional(),
    obstacleHeadings: z.array(z.array(z.number())).optional(),
    obstacleForecastX: z.array(z.array(z.array(z.number()))).optional(),
    obstacleForecastY: z.array(z.array(z.array(z.number()))).optional(),
    obstacleForecastHeading: z.array(z.array(z.array(z.number()))).optional(),
    obstacleForecastCovariance: z.array(z.array(z.array(z.array(z.array(z.number()))))).optional(),
    additionalPlots: z.array(AdditionalPlotSchema).optional(),
});

export type SimulationDataInput = z.input<typeof SimulationDataSchema>;
export type SimulationDataOutput = z.output<typeof SimulationDataSchema>;

export function parseSimulationData(data: unknown): SimulationDataOutput {
    return SimulationDataSchema.parse(data);
}
