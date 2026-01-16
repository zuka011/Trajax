import type { Theme } from "@/core/defaults";
import type { Plot } from "@/core/types";

export function getSeriesColor(series: Plot.Series, index: number, theme: Theme): string {
    return series.color ?? theme.colors.series[index % theme.colors.series.length];
}

export function getBoundValues(bound: Plot.Bound, timestepCount: number): number[] {
    if (typeof bound.values === "number") {
        return Array(timestepCount).fill(bound.values);
    }
    return bound.values;
}

export function computeYAxisRange(
    plots: Plot.Additional[],
    timestepCount: number,
): [number, number] {
    let minValue = Infinity;
    let maxValue = -Infinity;

    for (const plot of plots) {
        for (const series of plot.series) {
            for (const value of series.values) {
                minValue = Math.min(minValue, value);
                maxValue = Math.max(maxValue, value);
            }
        }

        if (plot.upperBound) {
            const values = getBoundValues(plot.upperBound, timestepCount);
            for (const value of values) {
                maxValue = Math.max(maxValue, value);
            }
        }

        if (plot.lowerBound) {
            const values = getBoundValues(plot.lowerBound, timestepCount);
            for (const value of values) {
                minValue = Math.min(minValue, value);
            }
        }

        if (plot.bands) {
            for (const band of plot.bands) {
                for (const value of band.lower) minValue = Math.min(minValue, value);
                for (const value of band.upper) maxValue = Math.max(maxValue, value);
            }
        }
    }

    const padding = (maxValue - minValue) * 0.1 || 0.1;
    return [minValue - padding, maxValue + padding];
}
