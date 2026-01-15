import { defaults } from "../core/defaults.js";
import { parseSimulationData } from "../core/schema.js";
import type { Visualizable } from "../core/types.js";
import styles from "../styles/main.css";
import { visualizerTemplate } from "../templates/base.js";
import browserScript from "./browser.bundle.js";

export function generate(rawData: unknown, title = defaults.title): string {
    const data = processData(parseSimulationData(rawData));

    return visualizerTemplate({
        title,
        styles: styles,
        data: JSON.stringify(data),
        script: browserScript,
    });
}

function processData(data: Visualizable.SimulationResult): Visualizable.ProcessedSimulationResult {
    return filterDistanceObstacles({ ...data, timestepCount: data.ego.x.length });
}

function filterDistanceObstacles(
    data: Visualizable.ProcessedSimulationResult,
    maxDistance: number = 1e6,
): Visualizable.ProcessedSimulationResult {
    const obstacles_data = data.obstacles;

    if (!obstacles_data) {
        return data;
    }

    const keepIndices = obstacles_data.x[0]
        .map((_, i) => i)
        .filter(
            (i) =>
                Math.abs(obstacles_data.x[0][i]) <= maxDistance &&
                Math.abs(obstacles_data.y[0][i]) <= maxDistance,
        );

    const pick = <T>(arr: T[]) => keepIndices.map((i) => arr[i]);

    const obstacles: Visualizable.Obstacles = {
        x: obstacles_data.x.map(pick),
        y: obstacles_data.y.map(pick),
        heading: obstacles_data.heading.map(pick),
    };

    if (obstacles_data.forecast) {
        const f = obstacles_data.forecast;
        obstacles.forecast = {
            x: f.x.map((h) => h.map(pick)),
            y: f.y.map((h) => h.map(pick)),
            heading: f.heading.map((h) => h.map(pick)),
            covariance: f.covariance?.map((h) => h.map((row) => row.map((col) => pick(col)))),
        };
    }

    return { ...data, obstacles };
}
