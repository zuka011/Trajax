import Plotly from "plotly.js-dist-min";
import type { Theme } from "@/core/defaults";
import type { Plot, Visualizable } from "@/core/types";
import { disableAutoRangeFor } from "@/utils/plots";
import type { VisualizationState } from "../state";
import type { UpdateManager } from "../update";
import { updaterCreator } from "./additional/factory";
import type {
    AdditionalPlotContext,
    AdditionalPlotTraceUpdater,
    AdditionalPlotUpdaterCreator,
} from "./additional/updater";
import { computeYAxisRange } from "./additional/utils";

const DEFAULT_TRACE_UPDATER_CREATORS: AdditionalPlotUpdaterCreator[] = [
    updaterCreator.seriesLines,
    updaterCreator.seriesMarkers,
    updaterCreator.bands,
    updaterCreator.bounds,
];

export interface PlotGroup {
    id: string;
    name: string;
    plots: Plot.Additional[];
}

interface AdditionalPlotConfig {
    container: HTMLElement;
    group: PlotGroup;
    data: Visualizable.ProcessedSimulationResult;
    state: VisualizationState;
    theme: Theme;
    updateManager: UpdateManager;
    traceUpdaterCreators?: AdditionalPlotUpdaterCreator[];
}

export function createAdditionalPlot(config: AdditionalPlotConfig): void {
    const updaterCreators = config.traceUpdaterCreators ?? DEFAULT_TRACE_UPDATER_CREATORS;
    const { container, group, data, state, theme, updateManager } = config;

    const times = data.ego.x.map((_, i) => i * data.info.timeStep);
    const allSeries = group.plots.flatMap((p) => p.series);
    const hasBands = group.plots.some((p) => p.bands && p.bands.length > 0);
    const yAxisLabel = group.plots[0]?.yAxisLabel ?? "";
    const yRange = computeYAxisRange(group.plots, data.timestepCount);
    const isLogScale = group.plots[0]?.yAxisScale === "log";

    const updaters: AdditionalPlotTraceUpdater[] = [];
    const allTraces: Plotly.Data[] = [];
    let currentIndex = 0;
    let seriesIndex = 0;

    const register = (factory: AdditionalPlotUpdaterCreator, context: AdditionalPlotContext) => {
        const updater = factory(context, currentIndex);
        const templates = updater.createTemplates(theme);
        updaters.push(updater);
        allTraces.push(...templates);
        currentIndex += templates.length;
    };

    for (const plot of group.plots) {
        const context: AdditionalPlotContext = {
            plot,
            times,
            timestepCount: data.timestepCount,
            seriesIndex,
            totalSeriesCount: allSeries.length,
        };

        for (const factory of updaterCreators) {
            register(factory, context);
        }

        seriesIndex += plot.series.length;
    }

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

    Plotly.newPlot(container, allTraces, layout, {
        scrollZoom: true,
        responsive: true,
        displayModeBar: "hover",
        displaylogo: false,
    }).then(() => disableAutoRangeFor(container));

    function render(): void {
        const xUpdates: Plotly.Datum[][] = [];
        const yUpdates: Plotly.Datum[][] = [];
        const indices: number[] = [];

        for (const updater of updaters) {
            const { data, updateIndices } = updater.updateTraces(state.currentTimestep);
            for (let i = 0; i < updateIndices.length; i++) {
                xUpdates.push(data[i].x);
                yUpdates.push(data[i].y);
                indices.push(updateIndices[i]);
            }
        }

        if (indices.length > 0) {
            Plotly.restyle(container, { x: xUpdates, y: yUpdates }, indices);
        }
    }

    render();
    updateManager.subscribe(render);
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
