import { noUpdater, type TraceUpdateCreator } from "./updater";

export const boundariesUpdater: TraceUpdateCreator = (data, index) => {
    const boundaries = data.boundaries;

    if (!boundaries) {
        return noUpdater(data, index);
    }

    return {
        createTemplates(theme) {
            return [boundaries.left, boundaries.right].map((it, index) => ({
                x: it.x,
                y: it.y,
                mode: "lines",
                line: { color: theme.colors.boundary, dash: "dash", width: 1 },
                name: "Boundaries",
                legendgroup: "boundaries",
                showlegend: index === 0,
            }));
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};
