import { laneEdge, laneSurfacePolygon } from "@/utils/math";
import type { Trace } from "../types";
import type { TraceUpdateCreator } from "./updater";
import { noUpdater } from "./updater";

export const roadNetworkUpdater: TraceUpdateCreator = (data, index) => {
    const network = data.network;

    if (!network || network.lanes.length === 0) {
        return noUpdater(data, index);
    }

    return {
        createTemplates(theme) {
            const surfaceX: (number | null)[] = [];
            const surfaceY: (number | null)[] = [];

            const solidMarkingsX: (number | null)[] = [];
            const solidMarkingsY: (number | null)[] = [];

            const dashedMarkingsX: (number | null)[] = [];
            const dashedMarkingsY: (number | null)[] = [];

            for (const lane of network.lanes) {
                const surface = laneSurfacePolygon(lane);
                surfaceX.push(...surface.x, null);
                surfaceY.push(...surface.y, null);

                const [left, right] = lane.markings;
                for (const [side, type] of [
                    ["left", left],
                    ["right", right],
                ] as const) {
                    if (type === "none") continue;
                    const edge = laneEdge(lane, side);
                    if (type === "dashed") {
                        dashedMarkingsX.push(...edge.x, null);
                        dashedMarkingsY.push(...edge.y, null);
                    } else {
                        solidMarkingsX.push(...edge.x, null);
                        solidMarkingsY.push(...edge.y, null);
                    }
                }
            }

            const traces: Trace[] = [
                {
                    x: surfaceX,
                    y: surfaceY,
                    mode: "lines",
                    fill: "toself",
                    fillcolor: theme.colors.road,
                    line: { width: 0 },
                    opacity: 0.8,
                    name: "Road",
                    hoverinfo: "skip",
                },
            ];

            if (solidMarkingsX.length > 0) {
                traces.push({
                    x: solidMarkingsX,
                    y: solidMarkingsY,
                    mode: "lines",
                    line: { color: theme.colors.roadMarking, width: 2 },
                    name: "Lane Marking",
                    legendgroup: "Lane Marking",
                    hoverinfo: "skip",
                });
            }

            if (dashedMarkingsX.length > 0) {
                traces.push({
                    x: dashedMarkingsX,
                    y: dashedMarkingsY,
                    mode: "lines",
                    line: { color: theme.colors.roadMarking, width: 2, dash: "dash" },
                    name: "Lane Marking",
                    showlegend: solidMarkingsX.length === 0,
                    legendgroup: "Lane Marking",
                    hoverinfo: "skip",
                });
            }

            return traces;
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};
