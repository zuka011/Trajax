import type { Plot } from "@/core/types";
import type { Trace } from "../types";
import { type AdditionalPlotUpdaterCreator, noUpdater } from "./updater";

function hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function computeBandWidth(band: Plot.Band): number {
    return Math.max(...band.upper) - Math.min(...band.lower);
}

export const bandsUpdater: AdditionalPlotUpdaterCreator = (context, index) => {
    const { plot, times } = context;
    const bands = plot.bands;

    if (!bands || bands.length === 0) {
        return noUpdater(context, index);
    }

    return {
        createTemplates(theme): Trace[] {
            const traces: Trace[] = [];
            const sortedBands = [...bands].sort(
                (a, b) => computeBandWidth(b) - computeBandWidth(a),
            );

            for (const band of sortedBands) {
                const color = band.color ?? theme.colors.bandColor;
                const borderColor = hexToRgba(color, 0.4);
                const fillColor = hexToRgba(color, 0.2);
                const legendGroup = `band-${band.label ?? "range"}`;

                traces.push({
                    x: times,
                    y: band.lower,
                    mode: "lines",
                    line: { color: borderColor, width: 1 },
                    legendgroup: legendGroup,
                    showlegend: false,
                    hoverinfo: "skip",
                });

                traces.push({
                    x: times,
                    y: band.upper,
                    mode: "lines",
                    fill: "tonexty",
                    fillcolor: fillColor,
                    line: { color: borderColor, width: 1 },
                    legendgroup: legendGroup,
                    name: band.label ?? "Range",
                    showlegend: !!band.label,
                });
            }

            return traces;
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};
