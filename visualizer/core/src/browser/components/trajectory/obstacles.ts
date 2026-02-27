import type { Visualizable } from "@/core/types";
import { write } from "@/utils/geometry";
import { noUpdater, type TraceUpdateCreator } from "./updater";

export const obstaclesUpdater: TraceUpdateCreator = (data, index) => {
	const obstacles = data.obstacles;

	if (!obstacles) {
		return noUpdater(data, index);
	}

	if (data.info.obstacleShape === "circle") {
		return circleObstaclesUpdater(data, index);
	}

	return boxObstaclesUpdater(data, index);
};

const boxObstaclesUpdater: TraceUpdateCreator = (data, index) => {
    const { wheelbase, vehicleWidth } = data.info;

    if (wheelbase == null || vehicleWidth == null) {
        throw new Error("The vehicle's wheelbase and width must be provided, when rendering obstacles as boxes.");
    }

	const obstacles = data.obstacles!;
	const maxObstacles = maxObstaclesIn(obstacles);
	const maxBufferLength = maxObstacles * write.box.pointCount + Math.max(0, maxObstacles - 1);
	const buffer = {
		x: new Array<number | null>(maxBufferLength),
		y: new Array<number | null>(maxBufferLength),
	};

	const updateBuffer = (t: number) => {
		const obstacleCount = obstacles.x[t]?.length ?? 0;
		let offset = 0;

		for (let i = 0; i < obstacleCount; i++) {
			const x = obstacles.x[t][i];
			const y = obstacles.y[t][i];
			const heading = obstacles.heading[t]?.[i];

			if (x == null || y == null || heading == null) {
				continue;
			}

			if (offset > 0) {
				buffer.x[offset] = null;
				buffer.y[offset] = null;
				offset++;
			}

			offset = write.box(x, y, heading, wheelbase, vehicleWidth, buffer, offset);
		}

		buffer.x.length = offset;
		buffer.y.length = offset;
	};

	return {
		createTemplates(theme) {
			updateBuffer(0);
			return [
				{
					x: buffer.x,
					y: buffer.y,
					mode: "lines" as const,
					fill: "toself" as const,
					fillcolor: theme.colors.obstacle,
					line: { color: theme.colors.obstacle, width: 2 },
					opacity: 0.9,
					name: "Obstacle",
					legendgroup: "obstacle",
					showlegend: true,
				},
			];
		},

		updateTraces(timeStep) {
			updateBuffer(timeStep);
			return {
				data: [buffer as { x: number[]; y: number[] }],
				updateIndices: [index],
			};
		},
	};
};

const circleObstaclesUpdater: TraceUpdateCreator = (data, index) => {
    if (data.info.obstacleRadius == null) {
        throw new Error("The obstacle radius must be provided, when rendering obstacles as circles.");
    }

	const obstacles = data.obstacles!;
	const radius = data.info.obstacleRadius;
	const maxObstacles = maxObstaclesIn(obstacles);

	const maxCircleBufferLength =
		maxObstacles * write.ellipse.pointCount + Math.max(0, maxObstacles - 1);
	const maxLineBufferLength =
		maxObstacles * write.headingLine.pointCount + Math.max(0, maxObstacles - 1);

	const circleBuffer = {
		x: new Array<number | null>(maxCircleBufferLength),
		y: new Array<number | null>(maxCircleBufferLength),
	};

	const lineBuffer = {
		x: new Array<number | null>(maxLineBufferLength),
		y: new Array<number | null>(maxLineBufferLength),
	};

	const updateBuffers = (t: number) => {
		const obstacleCount = obstacles.x[t]?.length ?? 0;
		let circleOffset = 0;
		let lineOffset = 0;

		for (let i = 0; i < obstacleCount; i++) {
			const x = obstacles.x[t][i];
			const y = obstacles.y[t][i];
			const heading = obstacles.heading[t]?.[i];

			if (x == null || y == null || heading == null) {
				continue;
			}

			if (circleOffset > 0) {
				circleBuffer.x[circleOffset] = null;
				circleBuffer.y[circleOffset] = null;
				circleOffset++;
				lineBuffer.x[lineOffset] = null;
				lineBuffer.y[lineOffset] = null;
				lineOffset++;
			}

			circleOffset = write.ellipse(x, y, radius, radius, 0, circleBuffer, circleOffset);
			lineOffset = write.headingLine(x, y, radius, heading, lineBuffer, lineOffset);
		}

		circleBuffer.x.length = circleOffset;
		circleBuffer.y.length = circleOffset;
		lineBuffer.x.length = lineOffset;
		lineBuffer.y.length = lineOffset;
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
					fillcolor: theme.colors.obstacle,
					line: { color: theme.colors.obstacle, width: 2 },
					opacity: 0.9,
					name: "Obstacle",
					legendgroup: "obstacle",
					showlegend: true,
				},
				{
					x: lineBuffer.x,
					y: lineBuffer.y,
					mode: "lines" as const,
					line: { color: theme.colors.obstacleBorder, width: 3 },
					name: "Obstacle",
					legendgroup: "obstacle",
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

function maxObstaclesIn(obstacles: Visualizable.Obstacles): number {
	let max = 0;

	for (const timestep of obstacles.x) {
		max = Math.max(max, timestep.length);
	}

	return max;
}
