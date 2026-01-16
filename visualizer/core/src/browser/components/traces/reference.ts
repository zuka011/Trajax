import type { TraceUpdateCreator } from "./updater";

export const referencePathUpdater: TraceUpdateCreator = (data, _index) => {
    return {
        createTemplates(theme) {
            return [
                {
                    x: data.reference.x,
                    y: data.reference.y,
                    mode: "lines",
                    line: { color: theme.colors.reference, dash: "dash", width: 2 },
                    name: "Reference",
                    legendgroup: "reference",
                },
            ];
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};
