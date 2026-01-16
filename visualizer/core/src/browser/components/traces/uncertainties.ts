import { defaults } from "@/core/defaults";
import type { Arrays } from "@/core/types";
import { covarianceToEllipse } from "@/utils/math";
import { noUpdater, type TraceUpdateCreator } from "./updater";

const ELLIPSE_SEGMENTS = 36;

export const uncertaintiesUpdater: TraceUpdateCreator = (data, index) => {
    const forecasts = data.obstacles?.forecast;
    const covariances = forecasts?.covariance;

    if (!forecasts || !covariances) {
        return noUpdater(data, index);
    }

    const maxUncertainties = maxUncertaintiesIn(covariances);
    const buffers = Array.from({ length: maxUncertainties }, () => ({
        x: new Array<number>(ELLIPSE_SEGMENTS + 1),
        y: new Array<number>(ELLIPSE_SEGMENTS + 1),
    }));

    const clearUnusedBuffers = (startIndex: number) => {
        for (let i = startIndex; i < maxUncertainties; i++) {
            buffers[i].x.length = 0;
            buffers[i].y.length = 0;
        }
    };

    const updateBuffers = (t: number) => {
        const cov = covariances[t];

        let uncertaintyIndex = 0;

        outer: for (let h = 0; h < cov.length; h++) {
            const obstacleCount = forecasts.x[t]?.[h]?.length ?? 0;

            for (let k = 0; k < obstacleCount; k++) {
                if (uncertaintyIndex >= maxUncertainties) break outer;

                const c00 = cov[h][0]?.[0]?.[k];
                const c01 = cov[h][0]?.[1]?.[k];
                const c10 = cov[h][1]?.[0]?.[k];
                const c11 = cov[h][1]?.[1]?.[k];
                const fx = forecasts.x[t][h]?.[k];
                const fy = forecasts.y[t][h]?.[k];

                if (
                    c00 == null ||
                    c01 == null ||
                    c10 == null ||
                    c11 == null ||
                    fx == null ||
                    fy == null
                ) {
                    buffers[uncertaintyIndex].x.length = 0;
                    buffers[uncertaintyIndex].y.length = 0;
                    uncertaintyIndex++;
                    continue;
                }

                const covMatrix: [[number, number], [number, number]] = [
                    [c00, c01],
                    [c10, c11],
                ];

                const ellipse = covarianceToEllipse(covMatrix, defaults.confidenceScale);
                generateEllipsePoints(
                    fx,
                    fy,
                    ellipse.width / 2,
                    ellipse.height / 2,
                    ellipse.angle,
                    buffers[uncertaintyIndex],
                );
                uncertaintyIndex++;
            }
        }

        clearUnusedBuffers(uncertaintyIndex);
    };

    return {
        createTemplates(theme) {
            updateBuffers(0);
            return buffers.map((buffer, i) => ({
                x: buffer.x,
                y: buffer.y,
                mode: "lines" as const,
                fill: "toself" as const,
                fillcolor: theme.colors.forecast,
                line: { color: theme.colors.forecast, width: 1 },
                opacity: 0.1,
                name: "Uncertainty",
                legendgroup: "uncertainty",
                showlegend: i === 0,
            }));
        },

        updateTraces(time_step) {
            updateBuffers(time_step);
            return {
                data: buffers,
                updateIndices: buffers.map((_, i) => index + i),
            };
        },
    };
};

function generateEllipsePoints(
    cx: number,
    cy: number,
    rx: number,
    ry: number,
    angle: number,
    out: { x: number[]; y: number[] },
): void {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    for (let i = 0; i <= ELLIPSE_SEGMENTS; i++) {
        const theta = (i / ELLIPSE_SEGMENTS) * 2 * Math.PI;
        const ex = rx * Math.cos(theta);
        const ey = ry * Math.sin(theta);
        out.x[i] = cx + ex * cos - ey * sin;
        out.y[i] = cy + ex * sin + ey * cos;
    }
    out.x.length = ELLIPSE_SEGMENTS + 1;
    out.y.length = ELLIPSE_SEGMENTS + 1;
}

function maxUncertaintiesIn(covariances: Arrays.ForecastCovariances): number {
    let max = 0;

    for (const timestep of covariances) {
        let count = 0;
        for (const horizon of timestep) {
            count += horizon?.[0]?.[0]?.length ?? 0;
        }
        max = Math.max(max, count);
    }

    return max;
}
