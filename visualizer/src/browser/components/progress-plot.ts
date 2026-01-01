import Plotly from "plotly.js-dist-min";
import type { Theme } from "../../core/defaults.js";
import type { ProcessedSimulationData } from "../../core/types.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";

export function createProgressPlot(
    container: HTMLElement,
    data: ProcessedSimulationData,
    state: VisualizationState,
    theme: Theme,
    updateManager: UpdateManager,
): void {
    const times = data.positions_x.map((_, i) => i * data.time_step);
    let initialized = false;

    const layout: Partial<Plotly.Layout> = {
        title: { text: "Path Progress", font: { size: 12 } },
        xaxis: {
            title: { text: "Time (s)", font: { size: 10 } },
            showgrid: true,
            gridcolor: "#e0e0e0",
            fixedrange: true,
            tickfont: { size: 9 },
        },
        yaxis: {
            title: { text: "Progress (m)", font: { size: 10 } },
            range: [0, data.path_length * 1.1],
            showgrid: true,
            gridcolor: "#e0e0e0",
            zeroline: true,
            zerolinecolor: "#0000006a",
            fixedrange: true,
            tickfont: { size: 9 },
        },
        margin: { t: 25, r: 10, b: 50, l: 50 },
        showlegend: false,
        hovermode: "closest",
    };

    function render(): void {
        const t = state.currentTimestep;
        const markerTime = t * data.time_step;
        const markerProgress = data.path_parameters[t];

        const traces: Plotly.Data[] = [
            {
                x: times,
                y: data.path_parameters,
                mode: "lines",
                line: { color: theme.colors.primary, width: 2 },
                name: "Progress",
                showlegend: false,
            },
            {
                x: times,
                y: times.map(() => data.path_length),
                mode: "lines",
                line: { color: theme.colors.secondary, dash: "dash", width: 2 },
                name: "Goal",
                showlegend: false,
            },
            {
                x: [markerTime],
                y: [markerProgress],
                mode: "markers",
                marker: { color: theme.colors.accent, size: 10 },
                name: "Current",
                showlegend: false,
            },
        ];

        if (!initialized) {
            Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
            initialized = true;
        } else {
            Plotly.react(container, traces, layout);
        }
    }

    render();
    updateManager.subscribe(render);
}
