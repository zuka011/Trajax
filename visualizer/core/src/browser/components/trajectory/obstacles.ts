import type { Visualizable } from "@/core/types";
import { noUpdater, type TraceUpdateCreator } from "./updater";

const VEHICLE_CORNER_COUNT = 5;
const LOCAL_CORNERS_X = [-0.5, 0.5, 0.5, -0.5, -0.5];
const LOCAL_CORNERS_Y = [-0.5, -0.5, 0.5, 0.5, -0.5];

export const obstaclesUpdater: TraceUpdateCreator = (data, index) => {
    const obstacles = data.obstacles;

    if (!obstacles) {
        return noUpdater(data, index);
    }

    const maxObstacles = maxObstaclesIn(obstacles);
    const buffers = Array.from({ length: maxObstacles }, () => ({
        x: new Array<number>(VEHICLE_CORNER_COUNT),
        y: new Array<number>(VEHICLE_CORNER_COUNT),
    }));

    const updateBuffers = (t: number) => {
        const obstacleCount = obstacles.x[t]?.length ?? 0;
        const { wheelbase, vehicleWidth } = data.info;

        for (let i = 0; i < maxObstacles; i++) {
            if (i < obstacleCount) {
                const x = obstacles.x[t][i];
                const y = obstacles.y[t][i];
                const heading = obstacles.heading[t]?.[i];

                if (x == null || y == null || heading == null) {
                    buffers[i].x.length = 0;
                    buffers[i].y.length = 0;
                    continue;
                }

                const cos = Math.cos(heading);
                const sin = Math.sin(heading);

                for (let j = 0; j < VEHICLE_CORNER_COUNT; j++) {
                    const dx = LOCAL_CORNERS_X[j] * wheelbase;
                    const dy = LOCAL_CORNERS_Y[j] * vehicleWidth;
                    buffers[i].x[j] = x + dx * cos - dy * sin;
                    buffers[i].y[j] = y + dx * sin + dy * cos;
                }

                buffers[i].x.length = VEHICLE_CORNER_COUNT;
                buffers[i].y.length = VEHICLE_CORNER_COUNT;
            } else {
                buffers[i].x.length = 0;
                buffers[i].y.length = 0;
            }
        }
    };

    return {
        createTemplates(theme) {
            updateBuffers(0);
            return buffers.map((buffer, i) => ({
                x: buffer.x,
                y: buffer.y,
                mode: "lines" as const,
                fill: "toself" as const,
                fillcolor: theme.colors.obstacle,
                line: { color: theme.colors.obstacle, width: 2 },
                opacity: 0.9,
                name: "Obstacle",
                legendgroup: "obstacle",
                showlegend: i === 0,
            }));
        },

        updateTraces(timeStep) {
            updateBuffers(timeStep);
            return {
                data: buffers,
                updateIndices: buffers.map((_, i) => index + i),
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
