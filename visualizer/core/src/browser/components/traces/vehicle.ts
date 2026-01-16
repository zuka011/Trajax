import type { TraceUpdateCreator } from "./updater";

const VEHICLE_CORNER_COUNT = 5;
const LOCAL_CORNERS_X = [-0.5, 0.5, 0.5, -0.5, -0.5];
const LOCAL_CORNERS_Y = [-0.5, -0.5, 0.5, 0.5, -0.5];

export const vehicleUpdater: TraceUpdateCreator = (data, index) => {
    const buffers = {
        x: new Array<number>(VEHICLE_CORNER_COUNT),
        y: new Array<number>(VEHICLE_CORNER_COUNT),
    };

    const updateBuffers = (t: number) => {
        const cos = Math.cos(data.ego.heading[t]);
        const sin = Math.sin(data.ego.heading[t]);
        const { wheelbase, vehicleWidth } = data.info;

        for (let i = 0; i < VEHICLE_CORNER_COUNT; i++) {
            const dx = LOCAL_CORNERS_X[i] * wheelbase;
            const dy = LOCAL_CORNERS_Y[i] * vehicleWidth;
            buffers.x[i] = data.ego.x[t] + dx * cos - dy * sin;
            buffers.y[i] = data.ego.y[t] + dx * sin + dy * cos;
        }
    };

    return {
        createTemplates(theme) {
            updateBuffers(0);
            return [
                {
                    x: buffers.x,
                    y: buffers.y,
                    mode: "lines",
                    fill: "toself",
                    fillcolor: theme.colors.accent,
                    line: { color: theme.colors.accent, width: 2 },
                    opacity: 0.9,
                    name: "Ego Vehicle",
                    legendgroup: "ego",
                },
            ];
        },

        updateTraces(time_step) {
            updateBuffers(time_step);
            return {
                data: [buffers],
                updateIndices: [index],
            };
        },
    };
};
