import { defaults } from "../core/defaults.js";
import { parseSimulationData } from "../core/schema.js";
import type { ProcessedSimulationData, SimulationData } from "../core/types.js";
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

function processData(data: SimulationData): ProcessedSimulationData {
    return { ...data, timestep_count: data.positions_x.length };
}
