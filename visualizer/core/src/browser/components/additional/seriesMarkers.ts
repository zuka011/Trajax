import type { Trace } from "../types";
import type { AdditionalPlotUpdaterCreator } from "./updater";
import { getSeriesColor } from "./utils";

export const seriesMarkersUpdater: AdditionalPlotUpdaterCreator = (context, index) => {
    const { plot, times, seriesIndex } = context;

    const buffers = plot.series.map(() => ({
        x: new Array<number>(1),
        y: new Array<number>(1),
    }));

    const updateBuffers = (currentTimestep: number) => {
        const markerTime = times[currentTimestep];
        plot.series.forEach((series, i) => {
            buffers[i].x[0] = markerTime;
            buffers[i].y[0] = series.values[currentTimestep];
        });
    };

    return {
        createTemplates(theme): Trace[] {
            updateBuffers(0);
            return plot.series.map((series, i) => {
                const color = getSeriesColor(series, seriesIndex + i, theme);
                return {
                    x: buffers[i].x,
                    y: buffers[i].y,
                    mode: "markers",
                    marker: { color, size: 10 },
                    showlegend: false,
                    legendgroup: series.label,
                    name: `Current ${series.label}`,
                };
            });
        },

        updateTraces(timeStep) {
            updateBuffers(timeStep);
            return {
                data: buffers,
                updateIndices: buffers.map((_, i) => index + i),
            };
        },
    };
};
