import type { Visualizable } from "@/core/types";
import { noUpdater, type TraceUpdateCreator } from "./updater";

export const forecastsUpdater: TraceUpdateCreator = (data, index) => {
    const forecasts = data.obstacles?.forecast;

    if (!forecasts) {
        return noUpdater(data, index);
    }

    const { maxForecasts, horizonLength } = forecastCountIn(forecasts);
    const buffers = Array.from({ length: maxForecasts }, () => ({
        x: new Array<number | null>(horizonLength),
        y: new Array<number | null>(horizonLength),
    }));

    const updateBuffers = (t: number) => {
        const forecast = data.obstacles?.forecast;
        const forecastCount = forecast?.x[t]?.[0]?.length ?? 0;
        const currentHorizon = forecast?.x[t]?.length ?? 0;

        for (let k = 0; k < maxForecasts; k++) {
            if (k < forecastCount) {
                for (let h = 0; h < currentHorizon; h++) {
                    buffers[k].x[h] = forecast!.x[t][h][k];
                    buffers[k].y[h] = forecast!.y[t][h][k];
                }
                buffers[k].x.length = currentHorizon;
                buffers[k].y.length = currentHorizon;
            } else {
                buffers[k].x.length = 0;
                buffers[k].y.length = 0;
            }
        }
    };

    return {
        createTemplates(theme) {
            updateBuffers(0);
            return buffers.map((buffer, i) => ({
                x: buffer.x,
                y: buffer.y,
                mode: "lines+markers" as const,
                line: { color: theme.colors.forecast, width: 2 },
                marker: { color: theme.colors.forecast, size: 4, symbol: "circle" },
                opacity: 0.6,
                name: "Forecast",
                legendgroup: "forecast",
                showlegend: i === 0,
            }));
        },

        updateTraces(time_step) {
            updateBuffers(time_step);
            return {
                data: buffers as { x: number[]; y: number[] }[],
                updateIndices: buffers.map((_, i) => index + i),
            };
        },
    };
};

function forecastCountIn(forecasts: Visualizable.ObstacleForecast): {
    maxForecasts: number;
    horizonLength: number;
} {
    let maxForecasts = 0;
    let horizonLength = 0;

    for (const timestep of forecasts.x) {
        horizonLength = Math.max(horizonLength, timestep.length);
        for (const horizon of timestep) {
            maxForecasts = Math.max(maxForecasts, horizon.length);
        }
    }

    return { maxForecasts, horizonLength };
}
