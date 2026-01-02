import Plotly from "plotly.js-dist-min";
import type {
    AdditionalPlot,
    PlotBound,
    PlotSeries,
    ProcessedSimulationData,
} from "../../core/types.js";
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
    }

    const padding = (maxValue - minValue) * 0.1 || 0.1;
    return [minValue - padding, maxValue + padding];
}

function getSeriesColor(series: PlotSeries, index: number): string {
    return series.color ?? DEFAULT_COLORS[index % DEFAULT_COLORS.length];
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
        const firstSeriesColor = getSeriesColor(plot.series[0], seriesIndex);

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

            traces.push({
                x: [markerTime],
                y: [series.values[currentTimestep]],
                mode: "markers",
                marker: { color, size: 10 },
                showlegend: false,
            });

            seriesIndex++;
        }

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
    const yAxisLabel = group.plots[0]?.yAxisLabel ?? "";
    const yRange = computeYAxisRange(group.plots, data.timestepCount);
    
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
            range: yRange,
            showgrid: true,
            gridcolor: "#e0e0e0",
            zeroline: true,
            zerolinecolor: "#0000006a",
            fixedrange: true,
            tickfont: { size: 9 },
        },
        margin: { t: 25, r: 10, b: 50, l: 50 },
        showlegend: allSeries.length > 1,
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
        const traces = buildTraces(group, times, data.timestepCount, state.currentTimestep);

        if (!initialized) {
            Plotly.newPlot(container, traces, layout, {
                responsive: true,
                displayModeBar: false,
            });
            initialized = true;
        } else {
            Plotly.react(container, traces, layout);
        }
    }

    render();
    updateManager.subscribe(render);
}
