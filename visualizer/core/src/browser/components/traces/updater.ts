import type { Theme } from "@/core/defaults";
import type { Visualizable } from "@/core/types";

export type Trace = Plotly.Data;

export interface TraceUpdates {
    /**
     * Reference to the data arrays containing the updated trace data.
     */
    data: { x: number[]; y: number[] }[];

    /**
     * The indices of the traces to be updated.
     */
    updateIndices: number[];
}

export interface TraceUpdater {
    /**
     * Creates template traces that will later be updated with new data.
     *
     * @param theme The current theme to style the traces.
     * @returns The trace templates.
     */
    createTemplates(theme: Theme): Trace[];

    /**
     * Updates the trace data for the current time step.
     *
     * @param time_step The current time step index.
     * @return The updated trace data.
     */
    updateTraces(time_step: number): TraceUpdates;
}

export type TraceUpdateCreator = (
    data: Visualizable.ProcessedSimulationResult,
    index: number,
) => TraceUpdater;

export const noUpdater: TraceUpdateCreator = (_data, _index) => {
    return {
        createTemplates(_theme) {
            return [];
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};
