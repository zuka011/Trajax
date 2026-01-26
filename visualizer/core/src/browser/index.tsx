import { render } from "preact";
import { theme as defaultTheme } from "../core/defaults.js";
import type { Visualizable } from "../core/types.js";
import { AdditionalPlotsContainer } from "./components/AdditionalPlotsContainer.js";
import { App } from "./components/App.js";
import { ControlsContainer } from "./components/ControlsContainer.js";
import { SidebarToggle } from "./components/SidebarToggle.js";
import { createTrajectoryPlot } from "./components/trajectory-plot.js";
import { createInitialState } from "./state.js";
import { createUpdateManager } from "./update.js";

declare global {
    interface Window {
        __SIMULATION_RESULT__: Visualizable.ProcessedSimulationResult;
    }
}

function requireElement(id: string): HTMLElement {
    const element = document.getElementById(id);
    if (!element) {
        throw new Error(`Element with id "${id}" not found`);
    }
    return element;
}

function initialize(): void {
    const data = window.__SIMULATION_RESULT__;

    if (!data) {
        console.error("No simulation data found");
        return;
    }

    const state = createInitialState();
    const theme = defaultTheme;
    const updateManager = createUpdateManager();

    render(<SidebarToggle />, requireElement("sidebar-toggle-root"));
    render(
        <App data={data} state={state} updateManager={updateManager} />,
        requireElement("app-root"),
    );
    render(
        <ControlsContainer data={data} state={state} theme={theme} updateManager={updateManager} />,
        requireElement("controls-root"),
    );
    render(
        <AdditionalPlotsContainer
            data={data}
            state={state}
            theme={theme}
            updateManager={updateManager}
        />,
        requireElement("additional-plots-root"),
    );

    requestAnimationFrame(() => {
        createTrajectoryPlot({
            container: requireElement("trajectory-plot"),
            data,
            state,
            theme,
            updateManager,
        });
        updateManager.notify();
    });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
} else {
    initialize();
}
