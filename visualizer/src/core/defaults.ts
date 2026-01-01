export interface ThemeColors {
    primary: string;
    secondary: string;
    accent: string;
    accentDark: string;
    reference: string;
    background: string;
    border: string;
    text: string;
    infoBg: string;
    infoBorder: string;
    obstacle: string;
    obstacleBorder: string;
    forecast: string;
}

export interface ThemeSizes {
    vehicleSize: number;
    lineWidth: number;
    markerSize: number;
    defaultVehicleWidth: number;
    defaultWheelbase: number;
}

export interface AnimationSpeed {
    label: string;
    ms: number;
}

export interface ThemeAnimation {
    speeds: AnimationSpeed[];
}

export interface Theme {
    colors: ThemeColors;
    sizes: ThemeSizes;
    animation: ThemeAnimation;
}

export interface Defaults {
    timeStep: number;
    maxError: number;
    errorLabel: string;
    vehicleType: "triangle" | "rectangle";
    wheelbase: number;
    vehicleWidth: number;
    plotWidth: number;
    plotHeight: number;
    title: string;
    confidenceScale: number;
}

export const theme: Theme = {
    colors: {
        primary: "#3498db",
        secondary: "#2ecc71",
        accent: "#e74c3c",
        accentDark: "#c0392b",
        reference: "#bdc3c7",
        background: "#fafafa",
        border: "#ffffff",
        text: "#2c3e50",
        infoBg: "#f8f9fa",
        infoBorder: "#dee2e6",
        obstacle: "#7f8c8d",
        obstacleBorder: "#5a6263",
        forecast: "#9b59b6",
    },
    sizes: {
        vehicleSize: 15,
        lineWidth: 3,
        markerSize: 10,
        defaultVehicleWidth: 1.2,
        defaultWheelbase: 2.5,
    },
    animation: {
        speeds: [
            { label: "0.5x", ms: 200 },
            { label: "1x", ms: 100 },
            { label: "2x", ms: 50 },
            { label: "4x", ms: 25 },
        ],
    },
};

export const defaults: Defaults = {
    timeStep: 0.1,
    maxError: 1.0,
    errorLabel: "Lateral Error",
    vehicleType: "triangle" as const,
    wheelbase: 2.5,
    vehicleWidth: 1.2,
    plotWidth: 800,
    plotHeight: 600,
    title: "Simulation Visualization",
    confidenceScale: 2.0,
};
