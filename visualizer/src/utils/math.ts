import { defaults } from "../core/defaults.js";
import type { EllipseParameters } from "../core/types.js";

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

function eigenvectorAngle(a: number, b: number, c: number, d: number, lambda: number): number {
    if (Math.abs(b) > 1e-10) {
        return Math.atan2(lambda - a, b);
    } else if (Math.abs(c) > 1e-10) {
        return Math.atan2(c, lambda - d);
    } else {
        return a >= d ? 0 : Math.PI / 2;
    }
}
