import Plotly from "plotly.js-dist-min";
import type { Plot, Visualizable } from "../../core/types.js";
import { withoutAutorange } from "../../utils/plot.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";

const DEFAULT_COLORS = [
    "#3498db",
    "#e74c3c",
    "#2ecc71",
    "#9b59b6",
    "#f39c12",
    "#1abc9c",
    "#e91e63",
    "#795548",
];

const DEFAULT_BAND_COLOR = "#3498db";

function hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export interface PlotGroup {
    id: string;
    name: string;
    plots: Plot.Additional[];
}

export function groupPlots(plots: Plot.Additional[]): PlotGroup[] {
    const groupMap = new Map<string, Plot.Additional[]>();
    const ungrouped: Plot.Additional[] = [];

    for (const plot of plots) {
        if (plot.group) {
            const existing = groupMap.get(plot.group) ?? [];
            existing.push(plot);
            groupMap.set(plot.group, existing);
        } else {
            ungrouped.push(plot);
        }
    }

    const groups: PlotGroup[] = [];

    for (const [groupId, groupPlots] of groupMap) {
        groups.push({
            id: groupId,
            name: groupPlots.map((p) => p.name).join(" & "),
            plots: groupPlots,
        });
    }

    for (const plot of ungrouped) {
        groups.push({
            id: plot.id,
            name: plot.name,
            plots: [plot],
        });
    }

    return groups;
}

function getBoundValues(bound: Plot.Bound, timestepCount: number): number[] {
    if (typeof bound.values === "number") {
        return Array(timestepCount).fill(bound.values);
    }
    return bound.values;
}

function computeYAxisRange(plots: Plot.Additional[], timestepCount: number): [number, number] {
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
                for (const value of band.lower) {
                    minValue = Math.min(minValue, value);
                }
                for (const value of band.upper) {
                    maxValue = Math.max(maxValue, value);
                }
            }
        }
    }

    const padding = (maxValue - minValue) * 0.1 || 0.1;
    return [minValue - padding, maxValue + padding];
}

function getSeriesColor(series: Plot.Series, index: number): string {
    return series.color ?? DEFAULT_COLORS[index % DEFAULT_COLORS.length];
}

function computeBandWidth(band: Plot.Band): number {
    return Math.max(...band.upper) - Math.min(...band.lower);
}

function buildBandTraces(plot: Plot.Additional, times: number[], traces: Plotly.Data[]): void {
    if (!plot.bands || plot.bands.length === 0) {
        return;
    }

    const sortedBands = [...plot.bands].sort((a, b) => computeBandWidth(b) - computeBandWidth(a));

    for (const band of sortedBands) {
        const color = band.color ?? DEFAULT_BAND_COLOR;
        const borderColor = hexToRgba(color, 0.4);
        const fillColor = hexToRgba(color, 0.2);
        const legendGroup = `band-${band.label ?? "range"}`;

        traces.push({
            x: times,
            y: band.lower,
            mode: "lines" as const,
            line: { color: borderColor, width: 1 },
            legendgroup: legendGroup,
            showlegend: false,
            hoverinfo: "skip" as const,
        });

        traces.push({
            x: times,
            y: band.upper,
            mode: "lines" as const,
            fill: "tonexty" as const,
            fillcolor: fillColor,
            line: { color: borderColor, width: 1 },
            legendgroup: legendGroup,
            name: band.label ?? "Range",
            showlegend: !!band.label,
        });
    }
}

function buildBoundsTraces(
    plot: Plot.Additional,
    times: number[],
    timestepCount: number,
    traces: Plotly.Data[],
): void {
    const firstSeries = plot.series[0];
    const color = getSeriesColor(firstSeries, 0);
    const legendGroup = firstSeries.label;

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
}

function buildSeriesTraces(
    plot: Plot.Additional,
    times: number[],
    {
        seriesIndex,
        seriesCount,
        markerTime,
        currentTimestep,
    }: { seriesIndex: number; seriesCount: number; markerTime: number; currentTimestep: number },
    traces: Plotly.Data[],
): number {
    plot.series.forEach((series, index) => {
        const color = getSeriesColor(series, seriesIndex + index);

        traces.push({
            x: times,
            y: series.values,
            mode: "lines",
            line: { color, width: 2 },
            name: series.label,
            showlegend: seriesCount > 1,
            legendgroup: series.label,
        });

        traces.push({
            x: [markerTime],
            y: [series.values[currentTimestep]],
            mode: "markers",
            marker: { color, size: 10 },
            showlegend: false,
            legendgroup: series.label,
            name: `Current ${series.label}`,
        });
    });

    return seriesIndex + plot.series.length;
}

function buildTraces(
    group: PlotGroup,
    times: number[],
    timestepCount: number,
    currentTimestep: number,
): Plotly.Data[] {
    const traces: Plotly.Data[] = [];
    const allSeries = group.plots.flatMap((plot) => plot.series);
    const markerTime = times[currentTimestep];

    let seriesIndex = 0;

    for (const plot of group.plots) {
        buildBandTraces(plot, times, traces);
        buildBoundsTraces(plot, times, timestepCount, traces);

        seriesIndex = buildSeriesTraces(
            plot,
            times,
            { seriesIndex, seriesCount: allSeries.length, markerTime, currentTimestep },
            traces,
        );
    }

    return traces;
}

export function createAdditionalPlot(
    container: HTMLElement,
    group: PlotGroup,
    data: Visualizable.ProcessedSimulationResult,
    state: VisualizationState,
    updateManager: UpdateManager,
): void {
    const times = data.ego.x.map((_, i) => i * data.info.timeStep);
    const allSeries = group.plots.flatMap((plot) => plot.series);
    const hasBands = group.plots.some((plot) => plot.bands && plot.bands.length > 0);
    const yAxisLabel = group.plots[0]?.yAxisLabel ?? "";
    const yRange = computeYAxisRange(group.plots, data.timestepCount);
    const isLogScale = group.plots[0]?.yAxisScale === "log";

    let initialized = false;

    const layout: Partial<Plotly.Layout> = {
        title: { text: group.name, font: { size: 12 } },
        xaxis: {
            title: { text: "Time (s)", font: { size: 10 } },
            showgrid: true,
            gridcolor: "#e0e0e0",
            tickfont: { size: 9 },
        },
        yaxis: {
            title: { text: yAxisLabel, font: { size: 10 } },
            type: isLogScale ? "log" : undefined,
            range: isLogScale ? undefined : yRange,
            showgrid: true,
            gridcolor: "#e0e0e0",
            zeroline: !isLogScale,
            zerolinecolor: "#0000006a",
            tickfont: { size: 9 },
        },
        margin: { t: 75, r: 10, b: 50, l: 50 },
        showlegend: allSeries.length > 1 || hasBands,
        legend: {
            x: 1,
            y: 1,
            xanchor: "right",
            yanchor: "top",
            font: { size: 9 },
            bgcolor: "rgba(255,255,255,0.8)",
        },
        hovermode: "closest",
        uirevision: "constant",
        autosize: true,
    };
    const reactLayout = withoutAutorange(layout);

    function render(): void {
        const traces = buildTraces(group, times, data.timestepCount, state.currentTimestep);

        if (!initialized) {
            Plotly.newPlot(container, traces, layout, {
                scrollZoom: true,
                responsive: true,
                displayModeBar: "hover",
                displaylogo: false,
            });
            initialized = true;
        } else {
            Plotly.react(container, traces, reactLayout);
        }
    }

    render();
    updateManager.subscribe(render);
}
