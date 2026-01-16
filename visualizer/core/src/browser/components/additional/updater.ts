import type { Theme } from "@/core/defaults";
import type { Plot } from "@/core/types";
import type { Trace } from "../types";

export interface AdditionalPlotTraceUpdates {
    /**
     * Reference to the data arrays containing the updated trace data.
     */
    data: { x: number[]; y: number[] }[];

    /**
     * The indices of the traces to be updated.
     */
    updateIndices: number[];
}

export interface AdditionalPlotTraceUpdater {
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
     * @param timeStep The current time step index.
     * @return The updated trace data.
     */
    updateTraces(timeStep: number): AdditionalPlotTraceUpdates;
}

export interface AdditionalPlotContext {
    plot: Plot.Additional;
    times: number[];
    timestepCount: number;
    seriesIndex: number;
    totalSeriesCount: number;
}

export type AdditionalPlotUpdaterCreator = (
    context: AdditionalPlotContext,
    index: number,
) => AdditionalPlotTraceUpdater;

export const noUpdater: AdditionalPlotUpdaterCreator = () => {
    return {
        createTemplates(_theme) {
            return [];
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};
