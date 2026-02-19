import type { Visualizable } from "@/core/types";
import { write } from "@/utils/geometry";
import { noUpdater, type TraceUpdateCreator } from "./updater";

export const obstaclesUpdater: TraceUpdateCreator = (data, index) => {
    const obstacles = data.obstacles;

    if (!obstacles) {
        return noUpdater(data, index);
    }

    const maxObstacles = maxObstaclesIn(obstacles);
    const maxBufferLength = maxObstacles * write.box.pointCount + Math.max(0, maxObstacles - 1);
    const buffer = {
        x: new Array<number | null>(maxBufferLength),
        y: new Array<number | null>(maxBufferLength),
    };

    const updateBuffer = (t: number) => {
        const obstacleCount = obstacles.x[t]?.length ?? 0;
        const { wheelbase, vehicleWidth } = data.info;
        let offset = 0;

        for (let i = 0; i < obstacleCount; i++) {
            const x = obstacles.x[t][i];
            const y = obstacles.y[t][i];
            const heading = obstacles.heading[t]?.[i];

            if (x == null || y == null || heading == null) {
                continue;
            }

            if (offset > 0) {
                buffer.x[offset] = null;
                buffer.y[offset] = null;
                offset++;
            }

            offset = write.box(x, y, heading, wheelbase, vehicleWidth, buffer, offset);
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
                    mode: "lines" as const,
                    fill: "toself" as const,
                    fillcolor: theme.colors.obstacle,
                    line: { color: theme.colors.obstacle, width: 2 },
                    opacity: 0.9,
                    name: "Obstacle",
                    legendgroup: "obstacle",
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

function maxObstaclesIn(obstacles: Visualizable.Obstacles): number {
    let max = 0;

    for (const timestep of obstacles.x) {
        max = Math.max(max, timestep.length);
    }

    return max;
}
