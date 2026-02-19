import { describe, expect, it } from "vitest";
import { write } from "@/utils/geometry";

describe("write.box", () => {
    it("writes corners of an axis-aligned box when heading is zero", () => {
        const out = { x: new Array(write.box.pointCount), y: new Array(write.box.pointCount) };

        write.box(0, 0, 0, 2, 1, out, 0);

        expect(out.x[0]).toBeCloseTo(-1);
        expect(out.y[0]).toBeCloseTo(-0.5);
        expect(out.x[1]).toBeCloseTo(1);
        expect(out.y[1]).toBeCloseTo(-0.5);
        expect(out.x[2]).toBeCloseTo(1);
        expect(out.y[2]).toBeCloseTo(0.5);
        expect(out.x[3]).toBeCloseTo(-1);
        expect(out.y[3]).toBeCloseTo(0.5);
    });

    it("closes the polygon by repeating the first corner", () => {
        const out = { x: new Array(write.box.pointCount), y: new Array(write.box.pointCount) };

        write.box(0, 0, 0, 2, 1, out, 0);

        expect(out.x[write.box.pointCount - 1]).toBeCloseTo(out.x[0]);
        expect(out.y[write.box.pointCount - 1]).toBeCloseTo(out.y[0]);
    });

    it("rotates corners around the center", () => {
        const out = { x: new Array(write.box.pointCount), y: new Array(write.box.pointCount) };

        write.box(0, 0, Math.PI / 2, 2, 2, out, 0);

        expect(out.x[0]).toBeCloseTo(1);
        expect(out.y[0]).toBeCloseTo(-1);
    });

    it("translates the box to the given center", () => {
        const out = { x: new Array(write.box.pointCount), y: new Array(write.box.pointCount) };

        write.box(10, 20, 0, 2, 1, out, 0);

        expect(out.x[0]).toBeCloseTo(9);
        expect(out.y[0]).toBeCloseTo(19.5);
    });

    it("writes at the specified buffer offset", () => {
        const size = write.box.pointCount + 3;
        const out = {
            x: new Array(size).fill(null),
            y: new Array(size).fill(null),
        };

        write.box(0, 0, 0, 1, 1, out, 3);

        expect(out.x[2]).toBeNull();
        expect(out.x[3]).not.toBeNull();
    });

    it("writes the correct number of points", () => {
        const size = write.box.pointCount + 2;
        const out = {
            x: new Array(size).fill(null),
            y: new Array(size).fill(null),
        };

        write.box(0, 0, 0, 1, 1, out, 0);

        for (let i = 0; i < write.box.pointCount; i++) {
            expect(out.x[i]).not.toBeNull();
        }
        expect(out.x[write.box.pointCount]).toBeNull();
    });

    it("returns the new offset after writing", () => {
        const out = { x: new Array(20), y: new Array(20) };

        const newOffset = write.box(0, 0, 0, 1, 1, out, 7);

        expect(newOffset).toBe(7 + write.box.pointCount);
    });
});

describe("write.ellipse", () => {
    it("writes a circle when both radii are equal", () => {
        const out = {
            x: new Array(write.ellipse.pointCount),
            y: new Array(write.ellipse.pointCount),
        };

        write.ellipse(0, 0, 3, 3, 0, out, 0);

        for (let i = 0; i < write.ellipse.pointCount; i++) {
            const distance = Math.sqrt(out.x[i] ** 2 + out.y[i] ** 2);
            expect(distance).toBeCloseTo(3, 4);
        }
    });

    it("writes points centered at the given position", () => {
        const out = {
            x: new Array(write.ellipse.pointCount),
            y: new Array(write.ellipse.pointCount),
        };

        write.ellipse(5, 10, 1, 1, 0, out, 0);

        const averageX =
            out.x.reduce((a: number, b: number) => a + b, 0) / write.ellipse.pointCount;
        const averageY =
            out.y.reduce((a: number, b: number) => a + b, 0) / write.ellipse.pointCount;

        expect(averageX).toBeCloseTo(5, 1);
        expect(averageY).toBeCloseTo(10, 1);
    });

    it("respects the semi-major and semi-minor radii", () => {
        const out = {
            x: new Array(write.ellipse.pointCount),
            y: new Array(write.ellipse.pointCount),
        };

        write.ellipse(0, 0, 4, 2, 0, out, 0);

        const maxAbsoluteX = Math.max(...out.x.map(Math.abs));
        const maxAbsoluteY = Math.max(...out.y.map(Math.abs));

        expect(maxAbsoluteX).toBeCloseTo(4, 4);
        expect(maxAbsoluteY).toBeCloseTo(2, 4);
    });

    it("closes the curve by repeating the first point", () => {
        const out = {
            x: new Array(write.ellipse.pointCount),
            y: new Array(write.ellipse.pointCount),
        };

        write.ellipse(0, 0, 1, 1, 0, out, 0);

        expect(out.x[write.ellipse.pointCount - 1]).toBeCloseTo(out.x[0]);
        expect(out.y[write.ellipse.pointCount - 1]).toBeCloseTo(out.y[0]);
    });

    it("rotates the ellipse by the given angle", () => {
        const out = {
            x: new Array(write.ellipse.pointCount),
            y: new Array(write.ellipse.pointCount),
        };

        write.ellipse(0, 0, 4, 1, Math.PI / 2, out, 0);

        const maxAbsoluteX = Math.max(...out.x.map(Math.abs));
        const maxAbsoluteY = Math.max(...out.y.map(Math.abs));

        expect(maxAbsoluteX).toBeCloseTo(1, 3);
        expect(maxAbsoluteY).toBeCloseTo(4, 3);
    });

    it("writes at the specified buffer offset", () => {
        const size = write.ellipse.pointCount + 5;
        const out = {
            x: new Array(size).fill(null),
            y: new Array(size).fill(null),
        };

        write.ellipse(0, 0, 1, 1, 0, out, 5);

        expect(out.x[4]).toBeNull();
        expect(out.x[5]).not.toBeNull();
    });

    it("writes the correct number of points", () => {
        const size = write.ellipse.pointCount + 2;
        const out = {
            x: new Array(size).fill(null),
            y: new Array(size).fill(null),
        };

        write.ellipse(0, 0, 1, 1, 0, out, 0);

        for (let i = 0; i < write.ellipse.pointCount; i++) {
            expect(out.x[i]).not.toBeNull();
        }
        expect(out.x[write.ellipse.pointCount]).toBeNull();
    });

    it("returns the new offset after writing", () => {
        const out = { x: new Array(100), y: new Array(100) };

        const newOffset = write.ellipse(0, 0, 1, 1, 0, out, 10);

        expect(newOffset).toBe(10 + write.ellipse.pointCount);
    });
});
