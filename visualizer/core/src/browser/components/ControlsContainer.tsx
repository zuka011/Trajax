import type { FunctionalComponent } from "preact";
import { useEffect, useState } from "preact/hooks";
import type { Theme } from "../../core/defaults.js";
import type { Visualizable } from "../../core/types.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";
import { Controls } from "./Controls.js";

interface ControlsContainerProps {
    data: Visualizable.ProcessedSimulationResult;
    theme: Theme;
    state: VisualizationState;
    updateManager: UpdateManager;
}

export const ControlsContainer: FunctionalComponent<ControlsContainerProps> = ({
    data,
    theme,
    state,
    updateManager,
}) => {
    const [, forceUpdate] = useState(0);

    const speeds = theme.animation.speeds;
    const timestepCount = data.timestepCount;

    const notifyUpdate = (timestep: number): void => {
        state.currentTimestep = timestep;
        updateManager.notify();
        forceUpdate((n) => n + 1);
    };

    const startAnimation = (): void => {
        state.animationInterval = window.setInterval(() => {
            state.currentTimestep =
                state.currentTimestep >= timestepCount - 1 ? 0 : state.currentTimestep + 1;
            updateManager.notify();
            forceUpdate((n) => n + 1);
        }, speeds[state.speedIndex].ms);
    };

    const stopAnimation = (): void => {
        if (state.animationInterval !== null) {
            clearInterval(state.animationInterval);
            state.animationInterval = null;
        }
    };

    const handlePlay = (): void => {
        if (state.isPlaying) {
            stopAnimation();
            state.isPlaying = false;
        } else {
            startAnimation();
            state.isPlaying = true;
        }
        forceUpdate((n) => n + 1);
    };

    const handleReset = (): void => {
        notifyUpdate(0);
    };

    const handleSpeedChange = (): void => {
        state.speedIndex = (state.speedIndex + 1) % speeds.length;

        if (state.isPlaying) {
            stopAnimation();
            startAnimation();
        }
        forceUpdate((n) => n + 1);
    };

    const handleSliderChange = (timestep: number): void => {
        if (state.isPlaying) {
            stopAnimation();
            state.isPlaying = false;
        }
        notifyUpdate(timestep);
    };

    useEffect(() => {
        const callback = (): void => {
            forceUpdate((n) => n + 1);
        };
        updateManager.subscribe(callback);
    }, []);

    useEffect(() => {
        return () => {
            stopAnimation();
        };
    }, []);

    return (
        <Controls
            isPlaying={state.isPlaying}
            speedIndex={state.speedIndex}
            currentTimestep={state.currentTimestep}
            timestepCount={timestepCount}
            speeds={speeds}
            onPlay={handlePlay}
            onReset={handleReset}
            onSpeedChange={handleSpeedChange}
            onSliderChange={handleSliderChange}
        />
    );
};
