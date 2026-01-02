import Plotly from "plotly.js-dist-min";
import type {
    AdditionalPlot,
    PlotBand,
    PlotBound,
    PlotSeries,
    ProcessedSimulationData,
} from "../../core/types.js";
import { applyLegendState, attachLegendHandler } from ".././legend-state.js";
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
    plots: AdditionalPlot[];
}

export function groupPlots(plots: AdditionalPlot[]): PlotGroup[] {
    const groupMap = new Map<string, AdditionalPlot[]>();
    const ungrouped: AdditionalPlot[] = [];

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

function getBoundValues(bound: PlotBound, timestepCount: number): number[] {
    if (typeof bound.values === "number") {
        return Array(timestepCount).fill(bound.values);
    }
    return bound.values;
}

function computeYAxisRange(plots: AdditionalPlot[], timestepCount: number): [number, number] {
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

function getSeriesColor(series: PlotSeries, index: number): string {
    return series.color ?? DEFAULT_COLORS[index % DEFAULT_COLORS.length];
}

function computeBandWidth(band: PlotBand): number {
    return Math.max(...band.upper) - Math.min(...band.lower);
}

function buildBandTraces(plot: AdditionalPlot, times: number[]): Plotly.Data[] {
    if (!plot.bands || plot.bands.length === 0) {
        return [];
    }

    const sortedBands = [...plot.bands].sort((a, b) => computeBandWidth(b) - computeBandWidth(a));
    const traces: Plotly.Data[] = [];

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

    return traces;
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

    for (const plot of group.plots) {
        traces.push(...buildBandTraces(plot, times));
    }

    let seriesIndex = 0;

    for (const plot of group.plots) {
        const firstSeriesColor = getSeriesColor(plot.series[0], seriesIndex);

        if (plot.upperBound) {
            traces.push({
                x: times,
                y: getBoundValues(plot.upperBound, timestepCount),
                mode: "lines",
                line: { color: firstSeriesColor, dash: "dash", width: 1 },
                name: plot.upperBound.label ?? "Upper Bound",
                showlegend: !!plot.upperBound.label,
            });
        }

        if (plot.lowerBound) {
            traces.push({
                x: times,
                y: getBoundValues(plot.lowerBound, timestepCount),
                mode: "lines",
                line: { color: firstSeriesColor, dash: "dash", width: 1 },
                name: plot.lowerBound.label ?? "Lower Bound",
                showlegend: !!plot.lowerBound.label,
            });
        }

        for (const series of plot.series) {
            const color = getSeriesColor(series, seriesIndex);

            traces.push({
                x: times,
                y: series.values,
                mode: "lines",
                line: { color, width: 2 },
                name: series.label,
                showlegend: allSeries.length > 1,
            });

            seriesIndex++;
        }
    }

    seriesIndex = 0;
    for (const plot of group.plots) {
        for (const series of plot.series) {
            const color = getSeriesColor(series, seriesIndex);

            traces.push({
                x: [markerTime],
                y: [series.values[currentTimestep]],
                mode: "markers",
                marker: { color, size: 10 },
                showlegend: false,
                legendgroup: series.label,
                name: `Current ${series.label}`,
            });

            seriesIndex++;
        }
    }

    return traces;
}

export function createAdditionalPlot(
    container: HTMLElement,
    group: PlotGroup,
    data: ProcessedSimulationData,
    state: VisualizationState,
    updateManager: UpdateManager,
): void {
    const times = data.positionsX.map((_, i) => i * data.timeStep);
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
            fixedrange: true,
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
            fixedrange: true,
            tickfont: { size: 9 },
        },
        margin: { t: 25, r: 10, b: 50, l: 50 },
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
        autosize: true,
    };

    function render(): void {
        const traces = applyLegendState(
            buildTraces(group, times, data.timestepCount, state.currentTimestep),
            group.id,
        );

        if (!initialized) {
            Plotly.newPlot(container, traces, layout, {
                responsive: true,
                displayModeBar: false,
            });
            attachLegendHandler(container, group.id, render);
            initialized = true;
        } else {
            Plotly.react(container, traces, layout);
        }
    }

    render();
    updateManager.subscribe(render);
}
