import type { Theme } from "@/core/defaults";
import type { Types } from "@/core/types";
import { laneEdge, laneSurfacePolygon } from "@/utils/math";
import type { Trace, TraceUpdateCreator } from "./updater";
import { noUpdater } from "./updater";

export const roadNetworkUpdater: TraceUpdateCreator = (data, _index) => {
    const network = data.network;

    if (!network || network.lanes.length === 0) {
        return noUpdater(data, _index);
    }

    return {
        createTemplates(theme) {
            const traces: Trace[] = [];
            let isFirstLane = true;
            let isFirstMarking = true;

            for (const lane of network.lanes) {
                const surface = laneSurfacePolygon(lane);
                traces.push({
                    x: surface.x,
                    y: surface.y,
                    mode: "lines",
                    fill: "toself",
                    fillcolor: theme.colors.road,
                    line: { color: theme.colors.road, width: 0 },
                    opacity: 0.8,
                    name: "Road",
                    legendgroup: "road",
                    showlegend: isFirstLane,
                    hoverinfo: "skip",
                });
                isFirstLane = false;

                const [leftMarking, rightMarking] = lane.markings;

                if (leftMarking !== "none") {
                    const edge = laneEdge(lane, "left");
                    traces.push(createMarkingTrace(edge, leftMarking, theme, isFirstMarking));
                    isFirstMarking = false;
                }

                if (rightMarking !== "none") {
                    const edge = laneEdge(lane, "right");
                    traces.push(createMarkingTrace(edge, rightMarking, theme, isFirstMarking));
                    isFirstMarking = false;
                }
            }

            return traces;
        },

        updateTraces() {
            return { data: [], updateIndices: [] };
        },
    };
};

function createMarkingTrace(
    edge: { x: number[]; y: number[] },
    markingType: Types.MarkingType,
    theme: Theme,
    showLegend: boolean,
): Trace {
    return {
        x: edge.x,
        y: edge.y,
        mode: "lines",
        line: {
            color: theme.colors.roadMarking,
            width: 2,
            dash: markingType === "dashed" ? "dash" : "solid",
        },
        name: "Lane Marking",
        legendgroup: "lane-marking",
        showlegend: showLegend,
        hoverinfo: "skip",
    };
}
