import type { Trace } from "../types";
import type { AdditionalPlotUpdaterCreator } from "./updater";
import { getSeriesColor } from "./utils";

export const seriesLinesUpdater: AdditionalPlotUpdaterCreator = (context) => {
    const { plot, times, seriesIndex, totalSeriesCount } = context;

    return {
        createTemplates(theme): Trace[] {
            return plot.series.map((series, i) => {
                const color = getSeriesColor(series, seriesIndex + i, theme);
                return {
                    x: times,
                    y: series.values,
                    mode: "lines",
                    line: { color, width: 2 },
                    name: series.label,
                    showlegend: totalSeriesCount > 1,
                    legendgroup: series.label,
                };
            });
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};
