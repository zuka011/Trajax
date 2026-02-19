import { write } from "@/utils/geometry";
import type { TraceUpdateCreator } from "./updater";

export const vehicleUpdater: TraceUpdateCreator = (data, index) => {
    const buffers = {
        x: new Array<number>(write.box.pointCount),
        y: new Array<number>(write.box.pointCount),
    };

    const updateBuffers = (t: number) => {
        const { wheelbase, vehicleWidth } = data.info;
        write.box(
            data.ego.x[t],
            data.ego.y[t],
            data.ego.heading[t],
            wheelbase,
            vehicleWidth,
            buffers,
            0,
        );
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

        updateTraces(timeStep) {
            updateBuffers(timeStep);
            return {
                data: [buffers],
                updateIndices: [index],
            };
        },
    };
};
