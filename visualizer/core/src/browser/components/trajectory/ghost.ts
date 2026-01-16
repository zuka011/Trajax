import type { TraceUpdateCreator } from "./updater";

export const ghostUpdater: TraceUpdateCreator = (data, index) => {
    const buffers = {
        x: new Array<number>(1),
        y: new Array<number>(1),
    };

    const updateBuffers = (t: number) => {
        if (data.ego.ghost?.x[t] !== undefined) {
            buffers.x[0] = data.ego.ghost.x[t];
            buffers.y[0] = data.ego.ghost.y[t];
            buffers.x.length = 1;
            buffers.y.length = 1;
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
                    mode: "markers",
                    marker: { color: theme.colors.secondary, size: 12, opacity: 0.5 },
                    name: "Ghost",
                    legendgroup: "ghost",
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
