import type { FunctionalComponent } from "preact";
import type { Theme } from "../../core/defaults.js";

interface ControlsProps {
    isPlaying: boolean;
    speedIndex: number;
    currentTimestep: number;
    timestepCount: number;
    speeds: Theme["animation"]["speeds"];
    onPlay: () => void;
    onReset: () => void;
    onSpeedChange: () => void;
    onSliderChange: (timestep: number) => void;
}

export const Controls: FunctionalComponent<ControlsProps> = ({
    isPlaying,
    speedIndex,
    currentTimestep,
    timestepCount,
    speeds,
    onPlay,
    onReset,
    onSpeedChange,
    onSliderChange,
}) => {
    const handleSliderInput = (e: Event): void => {
        const target = e.target as HTMLInputElement;
        onSliderChange(Number.parseInt(target.value, 10));
    };

    return (
        <div class="controls">
            <button
                class={`control-btn play-btn${isPlaying ? " paused" : ""}`}
                onClick={onPlay}
                type="button"
            >
                {isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <button class="control-btn reset-btn" onClick={onReset} type="button">
                ↺
            </button>
            <button class="control-btn speed-btn" onClick={onSpeedChange} type="button">
                {speeds[speedIndex].label}
            </button>
            <div class="slider-container">
                <input
                    type="range"
                    class="time-slider"
                    min="0"
                    max={timestepCount - 1}
                    value={currentTimestep}
                    onInput={handleSliderInput}
                />
            </div>
        </div>
    );
};
