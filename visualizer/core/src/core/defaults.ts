export interface ThemeColors {
    primary: string;
    secondary: string;
    accent: string;
    accentDark: string;
    reference: string;
    background: string;
    border: string;
    text: string;
    infoBackground: string;
    infoBorder: string;
    optimal: string;
    nominal: string;
    obstacle: string;
    obstacleBorder: string;
    forecast: string;
    road: string;
    roadMarking: string;
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
    vehicleType: "triangle" | "rectangle";
    wheelbase: number;
    vehicleWidth: number;
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
        infoBackground: "#f8f9fa",
        infoBorder: "#dee2e6",
        optimal: "#e63946",
        nominal: "#2a9d8f",
        obstacle: "#7f8c8d",
        obstacleBorder: "#5a6263",
        forecast: "#9b59b6",
        road: "#4a4a4a2d",
        roadMarking: "#00e5ff54",
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
    vehicleType: "triangle" as const,
    wheelbase: 2.5,
    vehicleWidth: 1.2,
    title: "Simulation Visualization",
    confidenceScale: 2.0,
};
