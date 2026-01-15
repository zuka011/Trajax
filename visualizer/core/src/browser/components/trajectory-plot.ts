import Plotly from "plotly.js-dist-min";
import { defaults, type Theme } from "../../core/defaults.js";
import type { Road, Types, Visualizable } from "../../core/types.js";
import { covarianceToEllipse, laneEdge, laneSurfacePolygon } from "../../utils/math.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";

type Trace = Plotly.Data;

interface TraceIndices {
    reference: number;
    actualPath: number;
    vehicle: number;
    ghost: number;
    optimalTrajectory: number;
    nominalTrajectory: number;
    obstaclesStart: number;
    forecastsStart: number;
    uncertaintyStart: number;
}

interface PreallocatedBuffers {
    vehicleCorners: number[];
    obstacleCorners: number[];
    ellipsePoints: number[];
    covarianceMatrix: [[number | null, number | null], [number | null, number | null]];
}

interface TraceDataArrays {
    actualPath: { x: number[]; y: number[] };
    vehicle: { x: number[]; y: number[] };
    ghost: { x: number[]; y: number[] };
    optimal: { x: number[]; y: number[] };
    nominal: { x: number[]; y: number[] };
    obstacles: { x: number[]; y: number[] }[];
    forecasts: { x: (number | null)[]; y: (number | null)[] }[];
    uncertainties: { x: number[]; y: number[] }[];
}

interface PlotContext {
    data: Visualizable.ProcessedSimulationResult;
    theme: Theme;
    indices: TraceIndices;
    buffers: PreallocatedBuffers;
    maxObstacles: number;
    maxForecasts: number;
    maxUncertainties: number;
    horizonLength: number;
    ellipseSegments: number;
    traceData: TraceDataArrays;
    updateIndices: number[];
    xUpdates: Plotly.Datum[][];
    yUpdates: Plotly.Datum[][];
}

const VEHICLE_CORNER_COUNT = 5;
const ELLIPSE_SEGMENTS = 36;

const LOCAL_CORNERS_X = [-0.5, 0.5, 0.5, -0.5, -0.5];
const LOCAL_CORNERS_Y = [-0.5, -0.5, 0.5, 0.5, -0.5];

function createPreallocatedBuffers(
    maxObstacles: number,
    ellipseSegments: number,
): PreallocatedBuffers {
    return {
        vehicleCorners: new Array(VEHICLE_CORNER_COUNT * 2),
        obstacleCorners: new Array(VEHICLE_CORNER_COUNT * 2 * maxObstacles),
        ellipsePoints: new Array((ellipseSegments + 1) * 2),
        covarianceMatrix: [
            [0, 0],
            [0, 0],
        ],
    };
}

function computeMaxCounts(data: Visualizable.ProcessedSimulationResult): {
    maxObstacles: number;
    maxForecasts: number;
    maxUncertainties: number;
    horizonLength: number;
} {
    let maxObstacles = 0;
    let maxForecasts = 0;
    let maxUncertainties = 0;
    let horizonLength = 0;

    if (data.obstacles?.x) {
        for (const timestepObstacles of data.obstacles.x) {
            maxObstacles = Math.max(maxObstacles, timestepObstacles.length);
        }
    }

    if (data.obstacles?.forecast?.x) {
        for (const timestepForecasts of data.obstacles.forecast.x) {
            horizonLength = Math.max(horizonLength, timestepForecasts.length);
            for (const horizon of timestepForecasts) {
                maxForecasts = Math.max(maxForecasts, horizon.length);
            }
        }
    }

    if (data.obstacles?.forecast?.covariance) {
        for (const timestepCov of data.obstacles.forecast.covariance) {
            let uncertaintiesThisTimestep = 0;
            for (const horizon of timestepCov) {
                const obstacleCount = horizon?.[0]?.[0]?.length ?? 0;
                uncertaintiesThisTimestep += obstacleCount;
            }
            maxUncertainties = Math.max(maxUncertainties, uncertaintiesThisTimestep);
        }
    }

    return { maxObstacles, maxForecasts, maxUncertainties, horizonLength };
}

export function createTrajectoryPlot(
    container: HTMLElement,
    data: Visualizable.ProcessedSimulationResult,
    state: VisualizationState,
    theme: Theme,
    updateManager: UpdateManager,
): void {
    let initialized = false;

    const { maxObstacles, maxForecasts, maxUncertainties, horizonLength } = computeMaxCounts(data);
    const buffers = createPreallocatedBuffers(maxObstacles, ELLIPSE_SEGMENTS);

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

    let context: PlotContext;

    function initialize(): void {
        const staticTraces = buildStaticTraces(data, theme);
        const { dynamicTraces, indices } = buildDynamicTraceTemplates(
            data,
            theme,
            maxObstacles,
            maxForecasts,
            maxUncertainties,
        );

        const traceData = createPreallocatedTraceData(
            data,
            maxObstacles,
            maxForecasts,
            maxUncertainties,
            horizonLength,
            ELLIPSE_SEGMENTS,
        );

        const updateIndices = createUpdateIndices(
            indices,
            maxObstacles,
            maxForecasts,
            maxUncertainties,
        );
        const { xUpdates, yUpdates } = createUpdateArrayReferences(
            traceData,
            maxObstacles,
            maxForecasts,
            maxUncertainties,
        );

        context = {
            data,
            theme,
            indices,
            buffers,
            maxObstacles,
            maxForecasts,
            maxUncertainties,
            horizonLength,
            ellipseSegments: ELLIPSE_SEGMENTS,
            traceData,
            updateIndices,
            xUpdates,
            yUpdates,
        };

        const allTraces = [...staticTraces, ...dynamicTraces];

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

        updateDynamicTraces(container, context, state.currentTimestep);
    }

    render();
    updateManager.subscribe(render);
}

function createPreallocatedTraceData(
    data: Visualizable.ProcessedSimulationResult,
    maxObstacles: number,
    maxForecasts: number,
    maxUncertainties: number,
    horizonLength: number,
    ellipseSegments: number,
): TraceDataArrays {
    const trajectoryLength = data.trajectories?.optimal?.x[0]?.length ?? 0;

    return {
        actualPath: {
            x: new Array<number>(data.ego.x.length),
            y: new Array<number>(data.ego.y.length),
        },
        vehicle: {
            x: new Array<number>(VEHICLE_CORNER_COUNT),
            y: new Array<number>(VEHICLE_CORNER_COUNT),
        },
        ghost: {
            x: new Array<number>(1),
            y: new Array<number>(1),
        },
        optimal: {
            x: new Array<number>(trajectoryLength),
            y: new Array<number>(trajectoryLength),
        },
        nominal: {
            x: new Array<number>(trajectoryLength),
            y: new Array<number>(trajectoryLength),
        },
        obstacles: Array.from({ length: maxObstacles }, () => ({
            x: new Array<number>(VEHICLE_CORNER_COUNT),
            y: new Array<number>(VEHICLE_CORNER_COUNT),
        })),
        forecasts: Array.from({ length: maxForecasts }, () => ({
            x: new Array<number | null>(horizonLength),
            y: new Array<number | null>(horizonLength),
        })),
        uncertainties: Array.from({ length: maxUncertainties }, () => ({
            x: new Array<number>(ellipseSegments + 1),
            y: new Array<number>(ellipseSegments + 1),
        })),
    };
}

function createUpdateIndices(
    indices: TraceIndices,
    maxObstacles: number,
    maxForecasts: number,
    maxUncertainties: number,
): number[] {
    const result: number[] = [
        indices.actualPath,
        indices.vehicle,
        indices.ghost,
        indices.optimalTrajectory,
        indices.nominalTrajectory,
    ];

    for (let i = 0; i < maxObstacles; i++) {
        result.push(indices.obstaclesStart + i);
    }
    for (let i = 0; i < maxForecasts; i++) {
        result.push(indices.forecastsStart + i);
    }
    for (let i = 0; i < maxUncertainties; i++) {
        result.push(indices.uncertaintyStart + i);
    }

    return result;
}

function createUpdateArrayReferences(
    traceData: TraceDataArrays,
    maxObstacles: number,
    maxForecasts: number,
    maxUncertainties: number,
): { xUpdates: Plotly.Datum[][]; yUpdates: Plotly.Datum[][] } {
    const xUpdates: Plotly.Datum[][] = [
        traceData.actualPath.x,
        traceData.vehicle.x,
        traceData.ghost.x,
        traceData.optimal.x,
        traceData.nominal.x,
    ];
    const yUpdates: Plotly.Datum[][] = [
        traceData.actualPath.y,
        traceData.vehicle.y,
        traceData.ghost.y,
        traceData.optimal.y,
        traceData.nominal.y,
    ];

    for (let i = 0; i < maxObstacles; i++) {
        xUpdates.push(traceData.obstacles[i].x);
        yUpdates.push(traceData.obstacles[i].y);
    }
    for (let i = 0; i < maxForecasts; i++) {
        xUpdates.push(traceData.forecasts[i].x);
        yUpdates.push(traceData.forecasts[i].y);
    }
    for (let i = 0; i < maxUncertainties; i++) {
        xUpdates.push(traceData.uncertainties[i].x);
        yUpdates.push(traceData.uncertainties[i].y);
    }

    return { xUpdates, yUpdates };
}

function buildStaticTraces(data: Visualizable.ProcessedSimulationResult, theme: Theme): Trace[] {
    const traces: Trace[] = [];

    if (data.network) {
        traces.push(...createRoadNetworkTraces(data.network, theme));
    }

    traces.push(createReferenceTrace(data, theme));

    return traces;
}

function buildDynamicTraceTemplates(
    data: Visualizable.ProcessedSimulationResult,
    theme: Theme,
    maxObstacles: number,
    maxForecasts: number,
    maxUncertainties: number,
): { dynamicTraces: Trace[]; indices: TraceIndices } {
    const traces: Trace[] = [];
    let staticTraceCount = 0;

    if (data.network) {
        for (const lane of data.network.lanes) {
            staticTraceCount += 1;
            const [leftMarking, rightMarking] = lane.markings;
            if (leftMarking !== "none") staticTraceCount += 1;
            if (rightMarking !== "none") staticTraceCount += 1;
        }
    }
    staticTraceCount += 1;

    let currentIndex = staticTraceCount;

    const indices: TraceIndices = {
        reference: staticTraceCount - 1,
        actualPath: currentIndex++,
        vehicle: currentIndex++,
        ghost: currentIndex++,
        optimalTrajectory: currentIndex++,
        nominalTrajectory: currentIndex++,
        obstaclesStart: currentIndex,
        forecastsStart: -1,
        uncertaintyStart: -1,
    };
    currentIndex += maxObstacles;
    indices.forecastsStart = currentIndex;
    currentIndex += maxForecasts;
    indices.uncertaintyStart = currentIndex;

    traces.push(createActualPathTemplate(theme));
    traces.push(createVehicleTemplate(theme));
    traces.push(createGhostTemplate(theme));
    traces.push(createOptimalTrajectoryTemplate());
    traces.push(createNominalTrajectoryTemplate());
    traces.push(...createObstacleTemplates(theme, maxObstacles));
    traces.push(...createForecastTemplates(theme, maxForecasts));
    traces.push(...createUncertaintyTemplates(theme, maxUncertainties));

    return { dynamicTraces: traces, indices };
}

function updateDynamicTraces(container: HTMLElement, context: PlotContext, t: number): void {
    const {
        data,
        buffers,
        maxObstacles,
        maxForecasts,
        maxUncertainties,
        ellipseSegments,
        traceData,
        updateIndices,
        xUpdates,
        yUpdates,
    } = context;

    updateActualPathData(data, t, traceData.actualPath);
    updateVehicleData(data, t, buffers.vehicleCorners, traceData.vehicle);

    updateGhostData(data, t, traceData.ghost);
    updateOptimalTrajectoryData(data, t, traceData.optimal);
    updateNominalTrajectoryData(data, t, traceData.nominal);

    updateObstaclesData(data, t, maxObstacles, buffers, traceData.obstacles);
    updateForecastsData(data, t, maxForecasts, traceData.forecasts);
    updateUncertaintyData(
        data,
        t,
        maxUncertainties,
        ellipseSegments,
        buffers,
        traceData.uncertainties,
    );

    const update: Partial<Plotly.PlotData> = {
        x: xUpdates as unknown as Plotly.Datum[],
        y: yUpdates as unknown as Plotly.Datum[],
    };

    Plotly.restyle(container, update, updateIndices);
}

function createActualPathTemplate(theme: Theme): Trace {
    return {
        x: [],
        y: [],
        mode: "lines",
        line: { color: theme.colors.primary, width: 2 },
        name: "Actual",
        legendgroup: "actual",
    };
}

function createVehicleTemplate(theme: Theme): Trace {
    return {
        x: [],
        y: [],
        mode: "lines",
        fill: "toself",
        fillcolor: theme.colors.accent,
        line: { color: theme.colors.accent, width: 2 },
        opacity: 0.9,
        name: "Ego Vehicle",
        legendgroup: "ego",
    };
}

function createGhostTemplate(theme: Theme): Trace {
    return {
        x: [],
        y: [],
        mode: "markers",
        marker: { color: theme.colors.secondary, size: 12, opacity: 0.5 },
        name: "Ghost",
        legendgroup: "ghost",
    };
}

function createOptimalTrajectoryTemplate(): Trace {
    return {
        x: [],
        y: [],
        mode: "lines+markers",
        line: { color: "#e63946", width: 2 },
        marker: { color: "#e63946", size: 5, symbol: "circle" },
        opacity: 0.8,
        name: "Optimal",
        legendgroup: "optimal",
    };
}

function createNominalTrajectoryTemplate(): Trace {
    return {
        x: [],
        y: [],
        mode: "lines+markers",
        line: { color: "#2a9d8f", width: 2 },
        marker: { color: "#2a9d8f", size: 5, symbol: "circle" },
        opacity: 0.8,
        name: "Nominal",
        legendgroup: "nominal",
    };
}

function createObstacleTemplates(theme: Theme, count: number): Trace[] {
    return Array.from({ length: count }, (_, i) => ({
        x: [],
        y: [],
        mode: "lines" as const,
        fill: "toself" as const,
        fillcolor: theme.colors.obstacle,
        line: { color: theme.colors.obstacle, width: 2 },
        opacity: 0.9,
        name: "Obstacle",
        legendgroup: "obstacle",
        showlegend: i === 0,
    }));
}

function createForecastTemplates(theme: Theme, count: number): Trace[] {
    return Array.from({ length: count }, (_, i) => ({
        x: [],
        y: [],
        mode: "lines+markers" as const,
        line: { color: theme.colors.forecast, width: 2 },
        marker: { color: theme.colors.forecast, size: 4, symbol: "circle" },
        opacity: 0.6,
        name: "Forecast",
        legendgroup: "forecast",
        showlegend: i === 0,
    }));
}

function createUncertaintyTemplates(theme: Theme, count: number): Trace[] {
    return Array.from({ length: count }, (_, i) => ({
        x: [],
        y: [],
        mode: "lines" as const,
        fill: "toself" as const,
        fillcolor: theme.colors.forecast,
        line: { color: theme.colors.forecast, width: 1 },
        opacity: 0.1,
        name: "Uncertainty",
        legendgroup: "uncertainty",
        showlegend: i === 0,
    }));
}

function updateActualPathData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    out: { x: number[]; y: number[] },
): void {
    const length = t + 1;
    for (let i = 0; i < length; i++) {
        out.x[i] = data.ego.x[i];
        out.y[i] = data.ego.y[i];
    }
    out.x.length = length;
    out.y.length = length;
}

function updateVehicleData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    buffer: number[],
    out: { x: number[]; y: number[] },
): void {
    transformCornersIntoBuffer(
        data.ego.x[t],
        data.ego.y[t],
        data.ego.heading[t],
        data.info.wheelbase,
        data.info.vehicleWidth,
        buffer,
        0,
    );

    for (let i = 0; i < VEHICLE_CORNER_COUNT; i++) {
        out.x[i] = buffer[i * 2];
        out.y[i] = buffer[i * 2 + 1];
    }
}

function updateGhostData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    out: { x: number[]; y: number[] },
): boolean {
    if (data.ego.ghost?.x[t] !== undefined) {
        out.x[0] = data.ego.ghost.x[t];
        out.y[0] = data.ego.ghost.y[t];
        out.x.length = 1;
        out.y.length = 1;
        return true;
    }
    out.x.length = 0;
    out.y.length = 0;
    return false;
}

function updateOptimalTrajectoryData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    out: { x: number[]; y: number[] },
): boolean {
    const optimal = data.trajectories?.optimal;
    if (optimal?.x[t]) {
        const source = optimal.x[t];
        const length = source.length;
        for (let i = 0; i < length; i++) {
            out.x[i] = source[i];
            out.y[i] = optimal.y[t][i];
        }
        out.x.length = length;
        out.y.length = length;
        return true;
    }
    out.x.length = 0;
    out.y.length = 0;
    return false;
}

function updateNominalTrajectoryData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    out: { x: number[]; y: number[] },
): boolean {
    const nominal = data.trajectories?.nominal;
    if (nominal?.x[t]) {
        const source = nominal.x[t];
        const length = source.length;
        for (let i = 0; i < length; i++) {
            out.x[i] = source[i];
            out.y[i] = nominal.y[t][i];
        }
        out.x.length = length;
        out.y.length = length;
        return true;
    }
    out.x.length = 0;
    out.y.length = 0;
    return false;
}

function updateObstaclesData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    maxObstacles: number,
    buffers: PreallocatedBuffers,
    out: { x: number[]; y: number[] }[],
): void {
    if (!data.obstacles) {
        throw new Error("No obstacle data available");
    }

    const obstacleCount = data.obstacles?.x[t]?.length ?? 0;

    for (let i = 0; i < maxObstacles; i++) {
        if (i < obstacleCount) {
            const offset = i * VEHICLE_CORNER_COUNT * 2;
            const [x, y, heading] = [
                data.obstacles.x[t][i],
                data.obstacles.y[t][i],
                data.obstacles.heading[t]?.[i],
            ];

            if (x == null || y == null || heading == null) {
                out[i].x.length = 0;
                out[i].y.length = 0;
                continue;
            }

            transformCornersIntoBuffer(
                x,
                y,
                heading,
                data.info.wheelbase,
                data.info.vehicleWidth,
                buffers.obstacleCorners,
                offset,
            );

            for (let j = 0; j < VEHICLE_CORNER_COUNT; j++) {
                out[i].x[j] = buffers.obstacleCorners[offset + j * 2];
                out[i].y[j] = buffers.obstacleCorners[offset + j * 2 + 1];
            }
            out[i].x.length = VEHICLE_CORNER_COUNT;
            out[i].y.length = VEHICLE_CORNER_COUNT;
        } else {
            out[i].x.length = 0;
            out[i].y.length = 0;
        }
    }
}

function updateForecastsData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    maxForecasts: number,
    out: { x: (number | null)[]; y: (number | null)[] }[],
): void {
    const forecast = data.obstacles?.forecast;
    const forecastCount = forecast?.x[t]?.[0]?.length ?? 0;
    const horizonLength = forecast?.x[t]?.length ?? 0;

    for (let k = 0; k < maxForecasts; k++) {
        if (k < forecastCount) {
            for (let h = 0; h < horizonLength; h++) {
                out[k].x[h] = forecast!.x[t][h][k];
                out[k].y[h] = forecast!.y[t][h][k];
            }
            out[k].x.length = horizonLength;
            out[k].y.length = horizonLength;
        } else {
            out[k].x.length = 0;
            out[k].y.length = 0;
        }
    }
}

function updateUncertaintyData(
    data: Visualizable.ProcessedSimulationResult,
    t: number,
    maxUncertainties: number,
    ellipseSegments: number,
    buffers: PreallocatedBuffers,
    out: { x: number[]; y: number[] }[],
): void {
    const forecast = data.obstacles?.forecast;
    const cov = forecast?.covariance?.[t];

    let uncertaintyIndex = 0;

    if (cov && forecast) {
        outer: for (let h = 0; h < cov.length; h++) {
            const obstacleCount = forecast.x[t]?.[h]?.length ?? 0;

            for (let k = 0; k < obstacleCount; k++) {
                if (uncertaintyIndex >= maxUncertainties) break outer;

                const c00 = cov[h][0]?.[0]?.[k];
                const c01 = cov[h][0]?.[1]?.[k];
                const c10 = cov[h][1]?.[0]?.[k];
                const c11 = cov[h][1]?.[1]?.[k];
                const fx = forecast.x[t][h]?.[k];
                const fy = forecast.y[t][h]?.[k];

                if (
                    c00 == null ||
                    c01 == null ||
                    c10 == null ||
                    c11 == null ||
                    fx == null ||
                    fy == null
                ) {
                    out[uncertaintyIndex].x.length = 0;
                    out[uncertaintyIndex].y.length = 0;
                    uncertaintyIndex++;
                    continue;
                }

                buffers.covarianceMatrix[0][0] = c00;
                buffers.covarianceMatrix[0][1] = c01;
                buffers.covarianceMatrix[1][0] = c10;
                buffers.covarianceMatrix[1][1] = c11;

                const ellipse = covarianceToEllipse(
                    buffers.covarianceMatrix as [[number, number], [number, number]],
                    defaults.confidenceScale,
                );

                generateEllipsePointsIntoBuffer(
                    fx,
                    fy,
                    ellipse.width / 2,
                    ellipse.height / 2,
                    ellipse.angle,
                    ellipseSegments,
                    buffers.ellipsePoints,
                );

                const pointCount = ellipseSegments + 1;
                for (let i = 0; i < pointCount; i++) {
                    out[uncertaintyIndex].x[i] = buffers.ellipsePoints[i * 2];
                    out[uncertaintyIndex].y[i] = buffers.ellipsePoints[i * 2 + 1];
                }
                out[uncertaintyIndex].x.length = pointCount;
                out[uncertaintyIndex].y.length = pointCount;
                uncertaintyIndex++;
            }
        }
    }

    while (uncertaintyIndex < maxUncertainties) {
        out[uncertaintyIndex].x.length = 0;
        out[uncertaintyIndex].y.length = 0;
        uncertaintyIndex++;
    }
}

function createRoadNetworkTraces(network: Road.Network, theme: Theme): Trace[] {
    const traces: Trace[] = [];
    let isFirstLane = true;
    let isFirstMarking = true;

    for (const lane of network.lanes) {
        const { surfaceTrace, markingTraces } = createLaneTraces(
            lane,
            theme,
            isFirstLane,
            isFirstMarking,
        );
        traces.push(surfaceTrace);
        traces.push(...markingTraces);
        isFirstLane = false;
        if (markingTraces.length > 0) {
            isFirstMarking = false;
        }
    }

    return traces;
}

function createLaneTraces(
    lane: Road.Lane,
    theme: Theme,
    showLaneLegend: boolean,
    showMarkingLegend: boolean,
): { surfaceTrace: Trace; markingTraces: Trace[] } {
    const surfacePolygon = laneSurfacePolygon(lane);
    const surfaceTrace: Trace = {
        x: surfacePolygon.x,
        y: surfacePolygon.y,
        mode: "lines",
        fill: "toself",
        fillcolor: theme.colors.road,
        line: { color: theme.colors.road, width: 0 },
        opacity: 0.8,
        name: "Road",
        legendgroup: "road",
        showlegend: showLaneLegend,
        hoverinfo: "skip",
    };

    const markingTraces: Trace[] = [];
    const [leftMarkingType, rightMarkingType] = lane.markings;

    if (leftMarkingType !== "none") {
        markingTraces.push(
            createMarkingTrace(laneEdge(lane, "left"), leftMarkingType, theme, showMarkingLegend),
        );
    }

    if (rightMarkingType !== "none") {
        markingTraces.push(
            createMarkingTrace(
                laneEdge(lane, "right"),
                rightMarkingType,
                theme,
                showMarkingLegend && leftMarkingType === "none",
            ),
        );
    }

    return { surfaceTrace, markingTraces };
}

function createMarkingTrace(
    edge: { x: number[]; y: number[] },
    markingType: Types.MarkingType,
    theme: Theme,
    showLegend: boolean,
): Trace {
    return {
        x: edge.x,
        y: edge.y,
        mode: "lines",
        line: {
            color: theme.colors.roadMarking,
            width: 2,
            dash: markingType === "dashed" ? "dash" : "solid",
        },
        name: "Lane Marking",
        legendgroup: "lane-marking",
        showlegend: showLegend,
        hoverinfo: "skip",
    };
}

function createReferenceTrace(data: Visualizable.ProcessedSimulationResult, theme: Theme): Trace {
    return {
        x: data.reference.x,
        y: data.reference.y,
        mode: "lines",
        line: { color: theme.colors.reference, dash: "dash", width: 2 },
        name: "Reference",
        legendgroup: "reference",
    };
}

function transformCornersIntoBuffer(
    x: number,
    y: number,
    heading: number,
    width: number,
    height: number,
    buffer: number[],
    offset: number = 0,
): void {
    const cos = Math.cos(heading);
    const sin = Math.sin(heading);

    for (let i = 0; i < VEHICLE_CORNER_COUNT; i++) {
        const dx = LOCAL_CORNERS_X[i] * width;
        const dy = LOCAL_CORNERS_Y[i] * height;
        buffer[offset + i * 2] = x + dx * cos - dy * sin;
        buffer[offset + i * 2 + 1] = y + dx * sin + dy * cos;
    }
}

function generateEllipsePointsIntoBuffer(
    cx: number,
    cy: number,
    rx: number,
    ry: number,
    angle: number,
    segments: number,
    buffer: number[],
): void {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * 2 * Math.PI;
        const ex = rx * Math.cos(theta);
        const ey = ry * Math.sin(theta);
        buffer[i * 2] = cx + ex * cos - ey * sin;
        buffer[i * 2 + 1] = cy + ex * sin + ey * cos;
    }
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
