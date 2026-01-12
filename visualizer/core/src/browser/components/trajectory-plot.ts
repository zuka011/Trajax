import Plotly from "plotly.js-dist-min";
import { defaults, type Theme } from "../../core/defaults.js";
import type { ProcessedSimulationData } from "../../core/types.js";
import { covarianceToEllipse, radiansToDegrees } from "../../utils/math.js";
import { withoutAutorange } from "../../utils/plot.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";

type Trace = Plotly.Data;

export function createTrajectoryPlot(
    container: HTMLElement,
    data: ProcessedSimulationData,
    state: VisualizationState,
    theme: Theme,
    updateManager: UpdateManager,
): void {
    let initialized = false;
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
    const reactLayout = withoutAutorange(layout);

    function render() {
        const t = state.currentTimestep;
        const traces = buildTraces(data, theme, t);

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

function buildTraces(data: ProcessedSimulationData, theme: Theme, t: number): Trace[] {
    const traces: Trace[] = [];

    traces.push(createReferenceTrace(data, theme));
    traces.push(createActualPathTrace(data, theme, t));
    traces.push(createVehicleTrace(data, theme, t));

    if (data.ego.ghost?.x[t] !== undefined) {
        traces.push(createGhostTrace(data, theme, t));
    }

    if (data.trajectories?.optimal?.x[t]) {
        traces.push(createOptimalTrajectoryTrace(data, t));
    }

    if (data.trajectories?.nominal?.x[t]) {
        traces.push(createNominalTrajectoryTrace(data, t));
    }

    if (data.obstacles?.x[t]) {
        traces.push(...createObstacleTraces(data, theme, t));
    }

    if (data.obstacles?.forecast?.x[t]?.length) {
        traces.push(...createForecastTraces(data, theme, t));
    }

    if (data.obstacles?.forecast?.covariance?.[t]) {
        traces.push(...createUncertaintyTraces(data, theme, t));
    }

    return traces;
}

function createReferenceTrace(data: ProcessedSimulationData, theme: Theme): Trace {
    return {
        x: data.reference.x,
        y: data.reference.y,
        mode: "lines",
        line: { color: theme.colors.reference, dash: "dash", width: 2 },
        name: "Reference",
        legendgroup: "reference",
    };
}

function createActualPathTrace(data: ProcessedSimulationData, theme: Theme, t: number): Trace {
    return {
        x: data.ego.x.slice(0, t + 1),
        y: data.ego.y.slice(0, t + 1),
        mode: "lines",
        line: { color: theme.colors.primary, width: 2 },
        name: "Actual",
        legendgroup: "actual",
    };
}

function createVehicleTrace(data: ProcessedSimulationData, theme: Theme, t: number): Trace {
    const corners = transformCorners(
        data.ego.x[t],
        data.ego.y[t],
        data.ego.heading[t],
        data.info.wheelbase,
        data.info.vehicleWidth,
    );
    return {
        x: corners.map((c) => c[0]),
        y: corners.map((c) => c[1]),
        mode: "lines",
        fill: "toself",
        fillcolor: theme.colors.accent,
        line: { color: theme.colors.accent, width: 2 },
        opacity: 0.9,
        name: "Ego Vehicle",
        legendgroup: "ego",
    };
}

function createGhostTrace(data: ProcessedSimulationData, theme: Theme, t: number): Trace {
    return {
        x: [data.ego.ghost!.x[t]],
        y: [data.ego.ghost!.y[t]],
        mode: "markers",
        marker: { color: theme.colors.secondary, size: 12, opacity: 0.5 },
        name: "Ghost",
        legendgroup: "ghost",
    };
}

function createOptimalTrajectoryTrace(data: ProcessedSimulationData, t: number): Trace {
    return {
        x: data.trajectories!.optimal!.x[t],
        y: data.trajectories!.optimal!.y[t],
        mode: "lines+markers",
        line: { color: "#e63946", width: 2 },
        marker: { color: "#e63946", size: 5, symbol: "circle" },
        opacity: 0.8,
        name: "Optimal",
        legendgroup: "optimal",
    };
}

function createNominalTrajectoryTrace(data: ProcessedSimulationData, t: number): Trace {
    return {
        x: data.trajectories!.nominal!.x[t],
        y: data.trajectories!.nominal!.y[t],
        mode: "lines+markers",
        line: { color: "#2a9d8f", width: 2 },
        marker: { color: "#2a9d8f", size: 5, symbol: "circle" },
        opacity: 0.8,
        name: "Nominal",
        legendgroup: "nominal",
    };
}

function createObstacleTraces(data: ProcessedSimulationData, theme: Theme, t: number): Trace[] {
    return data.obstacles!.x[t].map((ox, i) => {
        const corners = transformCorners(
            ox,
            data.obstacles!.y[t][i],
            data.obstacles!.heading[t]?.[i] ?? 0,
            data.info.wheelbase,
            data.info.vehicleWidth,
        );
        return {
            x: corners.map((c) => c[0]),
            y: corners.map((c) => c[1]),
            mode: "lines" as const,
            fill: "toself" as const,
            fillcolor: theme.colors.obstacle,
            line: { color: theme.colors.obstacle, width: 2 },
            opacity: 0.9,
            name: "Obstacle",
            legendgroup: "obstacle",
            showlegend: i === 0,
        };
    });
}

function createForecastTraces(data: ProcessedSimulationData, theme: Theme, t: number): Trace[] {
    const forecast = data.obstacles!.forecast!;
    const obstacleCount = forecast.x[t][0].length;
    return Array.from({ length: obstacleCount }, (_, k) => {
        const xPoints = forecast.x[t].map((h) => h[k]);
        const yPoints = forecast.y[t].map((h) => h[k]);
        const lastIdx = xPoints.length - 1;

        const lastX = xPoints[lastIdx];
        const prevX = xPoints[lastIdx - 1];
        const lastY = yPoints[lastIdx];
        const prevY = yPoints[lastIdx - 1];

        const dx = lastX !== null && prevX !== null ? lastX - prevX : 0;
        const dy = lastY !== null && prevY !== null ? lastY - prevY : 0;
        const angle = radiansToDegrees(Math.atan2(dy, dx));

        return {
            x: xPoints,
            y: yPoints,
            mode: "lines+markers" as const,
            line: { color: theme.colors.forecast, width: 2 },
            marker: {
                size: xPoints.map((_, i) => (i === lastIdx ? 10 : 4)),
                symbol: xPoints.map((_, i) => (i === lastIdx ? "triangle-up" : "circle")),
                angle: xPoints.map((_, i) => (i === lastIdx ? 90 - angle : 0)),
                color: theme.colors.forecast,
            },
            opacity: 0.6,
            name: "Forecast",
            legendgroup: "forecast",
            showlegend: k === 0,
        };
    });
}

function createUncertaintyTraces(data: ProcessedSimulationData, theme: Theme, t: number): Trace[] {
    const traces: Trace[] = [];
    const forecast = data.obstacles!.forecast!;
    const cov = forecast.covariance![t];
    const obstacleCount = cov[0][0][0].length;
    let isFirst = true;

    cov.forEach((_, h) => {
        for (let k = 0; k < obstacleCount; k++) {
            const c00 = cov[h][0][0][k];
            const c01 = cov[h][0][1][k];
            const c10 = cov[h][1][0][k];
            const c11 = cov[h][1][1][k];
            const fx = forecast.x[t][h][k];
            const fy = forecast.y[t][h][k];

            if (
                c00 === null ||
                c01 === null ||
                c10 === null ||
                c11 === null ||
                fx === null ||
                fy === null
            ) {
                continue;
            }

            const ellipse = covarianceToEllipse(
                [
                    [c00, c01],
                    [c10, c11],
                ],
                defaults.confidenceScale,
            );
            const points = generateEllipsePoints(
                fx,
                fy,
                ellipse.width / 2,
                ellipse.height / 2,
                ellipse.angle,
            );
            traces.push({
                x: points.map((p) => p[0]),
                y: points.map((p) => p[1]),
                mode: "lines",
                fill: "toself",
                fillcolor: theme.colors.forecast,
                line: { color: theme.colors.forecast, width: 1 },
                opacity: 0.1,
                name: "Uncertainty",
                legendgroup: "uncertainty",
                showlegend: isFirst,
            });
            isFirst = false;
        }
    });

    return traces;
}

function transformCorners(
    x: number,
    y: number,
    heading: number,
    width: number,
    height: number,
): number[][] {
    const cos = Math.cos(heading);
    const sin = Math.sin(heading);
    const halfW = width / 2;
    const halfH = height / 2;

    return [
        [-halfW, -halfH],
        [halfW, -halfH],
        [halfW, halfH],
        [-halfW, halfH],
        [-halfW, -halfH],
    ].map(([dx, dy]) => [x + dx * cos - dy * sin, y + dx * sin + dy * cos]);
}

function generateEllipsePoints(
    cx: number,
    cy: number,
    rx: number,
    ry: number,
    angle: number,
    segments = 36,
): number[][] {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    return Array.from({ length: segments + 1 }, (_, i) => {
        const theta = (i / segments) * 2 * Math.PI;
        const ex = rx * Math.cos(theta);
        const ey = ry * Math.sin(theta);
        return [cx + ex * cos - ey * sin, cy + ex * sin + ey * cos];
    });
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
