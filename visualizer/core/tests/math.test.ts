import { describe, expect, it } from "vitest";
import { covarianceToEllipse, laneEdge, laneSurfacePolygon, radiansToDegrees } from "@/utils/math";

describe("covarianceToEllipse", () => {
    it("returns a circle for an identity covariance matrix", () => {
        const result = covarianceToEllipse(
            [
                [1, 0],
                [0, 1],
            ],
            1,
        );

        expect(result.width).toBeCloseTo(2);
        expect(result.height).toBeCloseTo(2);
    });

    it("scales ellipse by the confidence factor", () => {
        const atOne = covarianceToEllipse(
            [
                [1, 0],
                [0, 1],
            ],
            1,
        );
        const atTwo = covarianceToEllipse(
            [
                [1, 0],
                [0, 1],
            ],
            2,
        );

        expect(atTwo.width).toBeCloseTo(atOne.width * 2);
        expect(atTwo.height).toBeCloseTo(atOne.height * 2);
    });

    it("returns an elongated ellipse for a diagonal covariance", () => {
        const result = covarianceToEllipse(
            [
                [4, 0],
                [0, 1],
            ],
            1,
        );

        expect(result.width).toBeGreaterThan(result.height);
    });

    it("aligns the ellipse along the dominant axis", () => {
        const horizontal = covarianceToEllipse(
            [
                [4, 0],
                [0, 1],
            ],
            1,
        );
        const vertical = covarianceToEllipse(
            [
                [1, 0],
                [0, 4],
            ],
            1,
        );

        expect(Math.abs(horizontal.angle)).toBeLessThan(0.01);
        expect(Math.abs(vertical.angle - Math.PI / 2)).toBeLessThan(0.01);
    });

    it("handles a zero covariance matrix", () => {
        const result = covarianceToEllipse(
            [
                [0, 0],
                [0, 0],
            ],
            1,
        );

        expect(result.width).toBe(0);
        expect(result.height).toBe(0);
    });
});

describe("radiansToDegrees", () => {
    it("converts zero radians to zero degrees", () => {
        expect(radiansToDegrees(0)).toBe(0);
    });

    it("converts pi radians to 180 degrees", () => {
        expect(radiansToDegrees(Math.PI)).toBeCloseTo(180);
    });

    it("converts pi/2 radians to 90 degrees", () => {
        expect(radiansToDegrees(Math.PI / 2)).toBeCloseTo(90);
    });

    it("converts 2*pi radians to 360 degrees", () => {
        expect(radiansToDegrees(2 * Math.PI)).toBeCloseTo(360);
    });
});

describe("laneEdge", () => {
    const straightLane = {
        x: [0, 1, 2],
        y: [0, 0, 0],
        boundaries: [1, 1] as [number, number],
        markings: ["solid", "solid"] as ["solid", "solid"],
    };

    it("offsets the left edge perpendicular to the path", () => {
        const edge = laneEdge(straightLane, "left");

        expect(edge.x).toHaveLength(3);
        for (const y of edge.y) {
            expect(y).toBeCloseTo(1);
        }
    });

    it("offsets the right edge perpendicular to the path", () => {
        const edge = laneEdge(straightLane, "right");

        expect(edge.x).toHaveLength(3);
        for (const y of edge.y) {
            expect(y).toBeCloseTo(-1);
        }
    });
});

describe("laneSurfacePolygon", () => {
    it("returns a closed polygon with left and reversed right edges", () => {
        const lane = {
            x: [0, 1, 2],
            y: [0, 0, 0],
            boundaries: [1, 1] as [number, number],
            markings: ["solid", "solid"] as ["solid", "solid"],
        };

        const polygon = laneSurfacePolygon(lane);
        const lastX = polygon.x[polygon.x.length - 1];
        const lastY = polygon.y[polygon.y.length - 1];

        expect(lastX).toBeCloseTo(polygon.x[0]);
        expect(lastY).toBeCloseTo(polygon.y[0]);
    });
});
