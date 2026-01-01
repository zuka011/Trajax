import { z } from "zod";

const ReferenceTrajectorySchema = z.object({
    x: z.array(z.number()),
    y: z.array(z.number()),
});

export const SimulationDataSchema = z.object({
    reference: ReferenceTrajectorySchema,
    positions_x: z.array(z.number()),
    positions_y: z.array(z.number()),
    headings: z.array(z.number()),
    path_parameters: z.array(z.number()),
    path_length: z.number(),
    time_step: z.number().default(0.1),
    errors: z.array(z.number()).optional(),
    ghost_x: z.array(z.number()).optional(),
    ghost_y: z.array(z.number()).optional(),
    max_error: z.number().default(1.0),
    error_label: z.string().default("Lateral Error"),
    vehicle_type: z.enum(["triangle", "car"]).default("triangle"),
    wheelbase: z.number().default(2.5),
    vehicle_width: z.number().default(1.2),
    obstacle_positions_x: z.array(z.array(z.number())).optional(),
    obstacle_positions_y: z.array(z.array(z.number())).optional(),
    obstacle_headings: z.array(z.array(z.number())).optional(),
    obstacle_forecast_x: z.array(z.array(z.array(z.number()))).optional(),
    obstacle_forecast_y: z.array(z.array(z.array(z.number()))).optional(),
    obstacle_forecast_heading: z.array(z.array(z.array(z.number()))).optional(),
    obstacle_forecast_covariance: z
        .array(z.array(z.array(z.array(z.array(z.number())))))
        .optional(),
});

export type SimulationDataInput = z.input<typeof SimulationDataSchema>;
export type SimulationDataOutput = z.output<typeof SimulationDataSchema>;

export function parseSimulationData(data: unknown): SimulationDataOutput {
    return SimulationDataSchema.parse(data);
}
