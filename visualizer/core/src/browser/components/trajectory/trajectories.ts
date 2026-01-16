import type { TraceUpdateCreator } from "./updater";

export const optimalTrajectoryUpdater: TraceUpdateCreator = (data, index) => {
    const trajectoryLength = data.trajectories?.optimal?.x[0]?.length ?? 0;

    const buffers = {
        x: new Array<number>(trajectoryLength),
        y: new Array<number>(trajectoryLength),
    };

    const updateBuffers = (t: number) => {
        const optimal = data.trajectories?.optimal;
        if (optimal?.x[t]) {
            const length = optimal.x[t].length;
            for (let i = 0; i < length; i++) {
                buffers.x[i] = optimal.x[t][i];
                buffers.y[i] = optimal.y[t][i];
            }
            buffers.x.length = length;
            buffers.y.length = length;
        } else {
            buffers.x.length = 0;
            buffers.y.length = 0;
        }
    };

    return {
        createTemplates(theme) {
            updateBuffers(0);
            return [
                {
                    x: buffers.x,
                    y: buffers.y,
                    mode: "lines+markers",
                    line: { color: theme.colors.optimal, width: 2 },
                    marker: { color: theme.colors.optimal, size: 5, symbol: "circle" },
                    opacity: 0.8,
                    name: "Optimal",
                    legendgroup: "optimal",
                },
            ];
        },

        updateTraces(timeStep: number) {
            updateBuffers(timeStep);
            return {
                data: [buffers],
                updateIndices: [index],
            };
        },
    };
};

export const nominalTrajectoryUpdater: TraceUpdateCreator = (data, index) => {
    const trajectoryLength = data.trajectories?.nominal?.x[0]?.length ?? 0;

    const buffers = {
        x: new Array<number>(trajectoryLength),
        y: new Array<number>(trajectoryLength),
    };

    const updateBuffers = (t: number) => {
        const nominal = data.trajectories?.nominal;
        if (nominal?.x[t]) {
            const length = nominal.x[t].length;
            for (let i = 0; i < length; i++) {
                buffers.x[i] = nominal.x[t][i];
                buffers.y[i] = nominal.y[t][i];
            }
            buffers.x.length = length;
            buffers.y.length = length;
        } else {
            buffers.x.length = 0;
            buffers.y.length = 0;
        }
    };

    return {
        createTemplates(theme) {
            updateBuffers(0);
            return [
                {
                    x: buffers.x,
                    y: buffers.y,
                    mode: "lines+markers",
                    line: { color: theme.colors.nominal, width: 2 },
                    marker: { color: theme.colors.nominal, size: 5, symbol: "circle" },
                    opacity: 0.8,
                    name: "Nominal",
                    legendgroup: "nominal",
                },
            ];
        },

        updateTraces(timeStep) {
            updateBuffers(timeStep);
            return {
                data: [buffers],
                updateIndices: [index],
            };
        },
    };
};
