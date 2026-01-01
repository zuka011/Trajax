import Plotly from "plotly.js-dist-min";
import type { Theme } from "../../core/defaults.js";
import type { ProcessedSimulationData } from "../../core/types.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";

export function createErrorPlot(
    container: HTMLElement,
    data: ProcessedSimulationData,
    state: VisualizationState,
    theme: Theme,
    updateManager: UpdateManager,
): void {
    if (!data.errors || data.errors.length === 0) {
        console.info("No error data available for error plot.");
        return;
    }

    let initialized = false;
    const times = data.positions_x.map((_, i) => i * data.time_step);
    const maxAbsError = Math.max(data.max_error * 1.2, ...data.errors.map(Math.abs));

    const layout: Partial<Plotly.Layout> = {
        title: { text: data.error_label || "Lateral Error", font: { size: 12 } },
        xaxis: {
            title: { text: "Time (s)", font: { size: 10 } },
            showgrid: true,
            gridcolor: "#e0e0e0",
            fixedrange: true,
            tickfont: { size: 9 },
        },
        yaxis: {
            title: { text: "Error (m)", font: { size: 10 } },
            range: [-maxAbsError, maxAbsError],
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
        const markerError = data.errors![t];

        const traces: Plotly.Data[] = [
            {
                x: times,
                y: data.errors,
                mode: "lines",
                line: { color: theme.colors.accent, width: 2 },
                name: "Error",
                showlegend: false,
            },
            {
                x: times,
                y: times.map(() => data.max_error),
                mode: "lines",
                line: { color: theme.colors.accentDark, dash: "dash", width: 1 },
                showlegend: false,
                name: "Max Error",
            },
            {
                x: times,
                y: times.map(() => -data.max_error),
                mode: "lines",
                line: { color: theme.colors.accentDark, dash: "dash", width: 1 },
                showlegend: false,
                name: "Min Error",
            },
            {
                x: [markerTime],
                y: [markerError],
                mode: "markers",
                marker: { color: theme.colors.accent, size: 10 },
                showlegend: false,
                name: "Current",
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
