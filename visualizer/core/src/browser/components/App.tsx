import type { FunctionalComponent } from "preact";
import { useEffect, useState } from "preact/hooks";
import type { ProcessedSimulationData } from "../../core/types.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";
import { InfoPanel } from "./InfoPanel.js";

interface AppProps {
    data: ProcessedSimulationData;
    state: VisualizationState;
    updateManager: UpdateManager;
}

export const App: FunctionalComponent<AppProps> = ({ data, state, updateManager }) => {
    const [, forceUpdate] = useState(0);

    useEffect(() => {
        const callback = (): void => {
            forceUpdate((n) => n + 1);
        };
        updateManager.subscribe(callback);
    }, []);

    return <InfoPanel currentTimestep={state.currentTimestep} data={data} />;
};
