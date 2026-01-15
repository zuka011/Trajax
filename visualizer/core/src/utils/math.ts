import { defaults } from "../core/defaults.js";
import type { EllipseParameters, Road } from "../core/types.js";

export function covarianceToEllipse(
    [[a, b], [c, d]]: [[number, number], [number, number]],
    confidence: number = defaults.confidenceScale,
): EllipseParameters {
    const trace = a + d;
    const det = a * d - b * c;
    const discriminant = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));

    const lambda1 = trace / 2 + discriminant;
    const lambda2 = trace / 2 - discriminant;

    return {
        width: 2 * Math.sqrt(Math.max(lambda1, 0)) * confidence,
        height: 2 * Math.sqrt(Math.max(lambda2, 0)) * confidence,
        angle: eigenvectorAngle(a, b, c, d, lambda1),
    };
}

export function radiansToDegrees(radians: number): number {
    return (radians * 180) / Math.PI;
}

export function laneSurfacePolygon(lane: Road.Lane): { x: number[]; y: number[] } {
    const leftEdge = laneEdge(lane, "left");
    const rightEdge = laneEdge(lane, "right");

    return {
        x: [...leftEdge.x, ...rightEdge.x.slice().reverse(), leftEdge.x[0]],
        y: [...leftEdge.y, ...rightEdge.y.slice().reverse(), leftEdge.y[0]],
    };
}

export function laneEdge(lane: Road.Lane, side: "left" | "right"): { x: number[]; y: number[] } {
    const width = side === "left" ? lane.boundaries[0] : lane.boundaries[1];
    const sign = side === "left" ? 1 : -1;
    const n = lane.x.length;
    const edgeX: number[] = [];
    const edgeY: number[] = [];

    for (let i = 0; i < n; i++) {
        const normals = normals_of(lane.x, lane.y, i);
        edgeX.push(lane.x[i] + sign * width * normals.x);
        edgeY.push(lane.y[i] + sign * width * normals.y);
    }

    return { x: edgeX, y: edgeY };
}

function normals_of(x: number[], y: number[], index: number): { x: number; y: number } {
    const n = x.length;

    let dx: number;
    let dy: number;

    if (index === 0) {
        dx = x[1] - x[0];
        dy = y[1] - y[0];
    } else if (index === n - 1) {
        dx = x[n - 1] - x[n - 2];
        dy = y[n - 1] - y[n - 2];
    } else {
        dx = x[index + 1] - x[index - 1];
        dy = y[index + 1] - y[index - 1];
    }

    const length = Math.sqrt(dx * dx + dy * dy);
    if (length < 1e-10) {
        return { x: 0, y: 1 };
    }

    return { x: -dy / length, y: dx / length };
}

function eigenvectorAngle(a: number, b: number, c: number, d: number, lambda: number): number {
    if (Math.abs(b) > 1e-10) {
        return Math.atan2(lambda - a, b);
    } else if (Math.abs(c) > 1e-10) {
        return Math.atan2(c, lambda - d);
    } else {
        return a >= d ? 0 : Math.PI / 2;
    }
}
