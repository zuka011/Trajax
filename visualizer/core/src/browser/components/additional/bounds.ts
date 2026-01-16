import type { Plot } from "@/core/types";
import type { Trace } from "../types";
import { type AdditionalPlotUpdaterCreator, noUpdater } from "./updater";
import { getSeriesColor } from "./utils";

export const boundsUpdater: AdditionalPlotUpdaterCreator = (context, index) => {
    const { plot, times, timestepCount, seriesIndex } = context;
    const first = plot.series[0];

    if (!first || (!plot.upperBound && !plot.lowerBound)) {
        return noUpdater(context, index);
    }

    return {
        createTemplates(theme): Trace[] {
            const traces: Trace[] = [];

            const color = getSeriesColor(first, seriesIndex, theme);
            const legendGroup = first.label;

            if (plot.upperBound) {
                traces.push({
                    x: times,
                    y: getBoundValues(plot.upperBound, timestepCount),
                    mode: "lines",
                    line: { color, dash: "dash", width: 1 },
                    name: plot.upperBound.label ?? "Upper Bound",
                    showlegend: !!plot.upperBound.label,
                    legendgroup: legendGroup,
                });
            }

            if (plot.lowerBound) {
                traces.push({
                    x: times,
                    y: getBoundValues(plot.lowerBound, timestepCount),
                    mode: "lines",
                    line: { color, dash: "dash", width: 1 },
                    name: plot.lowerBound.label ?? "Lower Bound",
                    showlegend: !!plot.lowerBound.label,
                    legendgroup: legendGroup,
                });
            }

            return traces;
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};

function getBoundValues(bound: Plot.Bound, timestepCount: number): number[] {
    if (typeof bound.values === "number") {
        return Array(timestepCount).fill(bound.values);
    }
    return bound.values;
}
