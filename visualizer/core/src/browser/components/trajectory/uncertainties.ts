import { defaults } from "@/core/defaults";
import type { Arrays } from "@/core/types";
import { withAlpha } from "@/utils/color";
import { write } from "@/utils/geometry";
import { covarianceToEllipse } from "@/utils/math";
import { noUpdater, type TraceUpdateCreator } from "./updater";

export const uncertaintiesUpdater: TraceUpdateCreator = (data, index) => {
    const forecasts = data.obstacles?.forecast;
    const covariances = forecasts?.covariance;

    if (!forecasts || !covariances) {
        return noUpdater(data, index);
    }

    const maxUncertainties = maxUncertaintiesIn(covariances);
    const maxBufferLength =
        maxUncertainties * write.ellipse.pointCount + Math.max(0, maxUncertainties - 1);
    const buffer = {
        x: new Array<number | null>(maxBufferLength),
        y: new Array<number | null>(maxBufferLength),
    };

    const updateBuffer = (t: number) => {
        const cov = covariances[t];
        let offset = 0;

        for (let h = 0; h < cov.length; h++) {
            const obstacleCount = forecasts.x[t]?.[h]?.length ?? 0;

            for (let k = 0; k < obstacleCount; k++) {
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
                    continue;
                }

                if (offset > 0) {
                    buffer.x[offset] = null;
                    buffer.y[offset] = null;
                    offset++;
                }

                const covMatrix: [[number, number], [number, number]] = [
                    [c00, c01],
                    [c10, c11],
                ];

                const ellipse = covarianceToEllipse(covMatrix, defaults.confidenceScale);

                offset = write.ellipse(
                    fx,
                    fy,
                    ellipse.width / 2,
                    ellipse.height / 2,
                    ellipse.angle,
                    buffer,
                    offset,
                );
            }
        }

        buffer.x.length = offset;
        buffer.y.length = offset;
    };

    return {
        createTemplates(theme) {
            updateBuffer(0);
            return [
                {
                    x: buffer.x,
                    y: buffer.y,
                    mode: "lines" as const,
                    fill: "toself" as const,
                    fillcolor: withAlpha(theme.colors.forecast, 0.08),
                    line: { color: withAlpha(theme.colors.forecast, 0.4), width: 1 },
                    name: "Uncertainty",
                    legendgroup: "uncertainty",
                    showlegend: true,
                },
            ];
        },

        updateTraces(timeStep) {
            updateBuffer(timeStep);
            return {
                data: [buffer as { x: number[]; y: number[] }],
                updateIndices: [index],
            };
        },
    };
};

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
