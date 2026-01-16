import Plotly from "plotly.js-dist-min";
import type { Theme } from "@/core/defaults";
import type { Visualizable } from "@/core/types";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";
import { updaterCreator } from "./traces/factory.js";
import type { Trace, TraceUpdateCreator, TraceUpdater } from "./traces/updater.js";

const DEFAULT_TRACE_UPDATER_CREATORS: TraceUpdateCreator[] = [
    updaterCreator.roadNetwork,
    updaterCreator.referencePath,
    updaterCreator.actualPath,
    updaterCreator.vehicle,
    updaterCreator.ghost,
    updaterCreator.optimalTrajectory,
    updaterCreator.nominalTrajectory,
    updaterCreator.obstacles,
    updaterCreator.obstacleForecasts,
    updaterCreator.forecastUncertainties,
];

interface PlotConfig {
    container: HTMLElement;
    data: Visualizable.ProcessedSimulationResult;
    state: VisualizationState;
    theme: Theme;
    updateManager: UpdateManager;
    traceUpdaterCreators?: TraceUpdateCreator[];
}

export function createTrajectoryPlot(config: PlotConfig): void {
    const updaterCreators = config.traceUpdaterCreators ?? DEFAULT_TRACE_UPDATER_CREATORS;
    const { container, data, state, theme, updateManager } = config;

    let initialized = false;
    let updaters: TraceUpdater[] = [];

    const layout: Partial<Plotly.Layout> = {
        title: { text: "Vehicle Trajectory" },
        xaxis: { ...createAxisConfig("X Position (m)"), scaleanchor: "y" },
        yaxis: createAxisConfig("Y Position (m)"),
        showlegend: true,
        dragmode: "pan",
        hovermode: "closest",
        uirevision: "constant",
        margin: { t: 60, r: 20, b: 60, l: 60 },
    };

    function initialize(): void {
        let currentIndex = 0;
        const allTraces: Trace[] = [];

        updaters = updaterCreators.map((factory) => {
            const updater = factory(data, currentIndex);
            const templates = updater.createTemplates(theme);
            allTraces.push(...templates);
            currentIndex += templates.length;
            return updater;
        });

        Plotly.newPlot(container, allTraces, layout, {
            scrollZoom: true,
            responsive: true,
            displayModeBar: "hover",
            displaylogo: false,
        });

        initialized = true;
    }
    function render(): void {
        if (!initialized) {
            initialize();
        }

        const xUpdates: Plotly.Datum[][] = [];
        const yUpdates: Plotly.Datum[][] = [];
        const allIndices: number[] = [];

        for (const updater of updaters) {
            const { data, updateIndices } = updater.updateTraces(state.currentTimestep);
            for (let i = 0; i < updateIndices.length; i++) {
                xUpdates.push(data[i].x);
                yUpdates.push(data[i].y);
                allIndices.push(updateIndices[i]);
            }
        }

        if (allIndices.length > 0) {
            Plotly.restyle(container, { x: xUpdates, y: yUpdates }, allIndices);
        }
    }

    render();
    updateManager.subscribe(render);
}

function createAxisConfig(title: string): Partial<Plotly.LayoutAxis> {
    return {
        title: { text: title },
        showline: false,
        showgrid: true,
        gridcolor: "#e0e0e0",
        zeroline: false,
    };
}
