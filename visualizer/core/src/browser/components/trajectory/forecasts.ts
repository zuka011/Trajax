import type { Visualizable } from "@/core/types";
import { noUpdater, type TraceUpdateCreator } from "./updater";

export const forecastsUpdater: TraceUpdateCreator = (data, index) => {
    const forecasts = data.obstacles?.forecast;

    if (!forecasts) {
        return noUpdater(data, index);
    }

    const { maxForecasts, horizonLength } = forecastCountIn(forecasts);
    const maxBufferLength = maxForecasts * horizonLength + Math.max(0, maxForecasts - 1);
    const buffer = {
        x: new Array<number | null>(maxBufferLength),
        y: new Array<number | null>(maxBufferLength),
    };

    const updateBuffer = (t: number) => {
        const forecast = data.obstacles?.forecast;
        const forecastCount = forecast?.x[t]?.[0]?.length ?? 0;
        const currentHorizon = forecast?.x[t]?.length ?? 0;
        let offset = 0;

        for (let k = 0; k < forecastCount; k++) {
            if (offset > 0) {
                buffer.x[offset] = null;
                buffer.y[offset] = null;
                offset++;
            }

            for (let h = 0; h < currentHorizon; h++) {
                buffer.x[offset] = forecast!.x[t][h][k];
                buffer.y[offset] = forecast!.y[t][h][k];
                offset++;
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
                    mode: "lines+markers" as const,
                    line: { color: theme.colors.forecast, width: 2 },
                    marker: { color: theme.colors.forecast, size: 4, symbol: "circle" },
                    opacity: 0.6,
                    name: "Forecast",
                    legendgroup: "forecast",
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
