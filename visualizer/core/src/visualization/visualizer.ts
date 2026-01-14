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
    return { ...data, timestepCount: data.ego.x.length };
}
