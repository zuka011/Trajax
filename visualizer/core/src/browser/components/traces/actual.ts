import type { TraceUpdateCreator } from "./updater";

export const actualPathUpdater: TraceUpdateCreator = (data, index) => {
    const source = data.ego;
    const buffers = {
        x: new Array<number>(source.x.length),
        y: new Array<number>(source.y.length),
    };

    const updateBuffers = (time_step: number) => {
        const length = time_step + 1;
        for (let i = 0; i < length; i++) {
            buffers.x[i] = source.x[i];
            buffers.y[i] = source.y[i];
        }
        buffers.x.length = length;
        buffers.y.length = length;
    };

    return {
        createTemplates(theme) {
            updateBuffers(0);
            return [
                {
                    x: buffers.x,
                    y: buffers.y,
                    mode: "lines",
                    line: { color: theme.colors.primary, width: 2 },
                    name: "Actual",
                    legendgroup: "actual",
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
