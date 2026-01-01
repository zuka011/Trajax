export interface VisualizationState {
    currentTimestep: number;
    isPlaying: boolean;
    animationInterval: number | null;
    speedIndex: number;
    visibility: Record<string, boolean>;
}

export function createInitialState(): VisualizationState {
    return {
        currentTimestep: 0,
        isPlaying: false,
        animationInterval: null,
        speedIndex: 2,
        visibility: {
            reference: true,
            actualPath: true,
            vehicle: true,
            ghost: true,
            obstacles: true,
            forecasts: true,
            uncertainty: true,
        },
    };
}
