export interface VisualizationState {
    currentTimestep: number;
    isPlaying: boolean;
    animationInterval: number | null;
    speedIndex: number;
}

export function createInitialState(): VisualizationState {
    return {
        currentTimestep: 0,
        isPlaying: false,
        animationInterval: null,
        speedIndex: 2,
    };
}
