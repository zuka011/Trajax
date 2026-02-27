import { write } from "@/utils/geometry";
import type { TraceUpdateCreator } from "./updater";

export const vehicleUpdater: TraceUpdateCreator = (data, index) => {
	if (data.info.vehicleType === "circle") {
		return circleVehicleUpdater(data, index);
	}

	return boxVehicleUpdater(data, index);
};

const boxVehicleUpdater: TraceUpdateCreator = (data, index) => {
    const { wheelbase, vehicleWidth } = data.info;

    if (wheelbase == null || vehicleWidth == null) {
        throw new Error("The vehicle's wheelbase and width must be provided, when rendering the ego vehicle as a box.");
    }

	const buffers = {
		x: new Array<number>(write.box.pointCount),
		y: new Array<number>(write.box.pointCount),
	};

	const updateBuffers = (t: number) => {
		write.box(
			data.ego.x[t],
			data.ego.y[t],
			data.ego.heading[t],
			wheelbase,
			vehicleWidth,
			buffers,
			0,
		);
	};

	return {
		createTemplates(theme) {
			updateBuffers(0);
			return [
				{
					x: buffers.x,
					y: buffers.y,
					mode: "lines",
					fill: "toself",
					fillcolor: theme.colors.accent,
					line: { color: theme.colors.accent, width: 2 },
					opacity: 0.9,
					name: "Ego Vehicle",
					legendgroup: "ego",
				},
			];
		},

		updateTraces(timeStep) {
			updateBuffers(timeStep);
			return {
				data: [buffers],
				updateIndices: [index],
			};
		},
	};
};

const circleVehicleUpdater: TraceUpdateCreator = (data, index) => {
    if (data.info.vehicleRadius == null) {
        throw new Error("The vehicle's radius must be provided, when rendering the ego vehicle as a circle.");
    }

	const radius = data.info.vehicleRadius;

	const circleBuffer = {
		x: new Array<number | null>(write.ellipse.pointCount),
		y: new Array<number | null>(write.ellipse.pointCount),
	};

	const lineBuffer = {
		x: new Array<number | null>(write.headingLine.pointCount),
		y: new Array<number | null>(write.headingLine.pointCount),
	};

	const updateBuffers = (t: number) => {
		write.ellipse(data.ego.x[t], data.ego.y[t], radius, radius, 0, circleBuffer, 0);
		write.headingLine(data.ego.x[t], data.ego.y[t], radius, data.ego.heading[t], lineBuffer, 0);
	};

	return {
		createTemplates(theme) {
			updateBuffers(0);
			return [
				{
					x: circleBuffer.x,
					y: circleBuffer.y,
					mode: "lines" as const,
					fill: "toself" as const,
					fillcolor: theme.colors.accent,
					line: { color: theme.colors.accent, width: 2 },
					opacity: 0.9,
					name: "Ego Vehicle",
					legendgroup: "ego",
				},
				{
					x: lineBuffer.x,
					y: lineBuffer.y,
					mode: "lines" as const,
					line: { color: theme.colors.accentDark, width: 3 },
					name: "Ego Vehicle",
					legendgroup: "ego",
					showlegend: false,
				},
			];
		},

		updateTraces(timeStep) {
			updateBuffers(timeStep);
			return {
				data: [
					circleBuffer as { x: number[]; y: number[] },
					lineBuffer as { x: number[]; y: number[] },
				],
				updateIndices: [index, index + 1],
			};
		},
	};
};
