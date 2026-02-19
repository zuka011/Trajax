import { describe, expect, it } from "vitest";
import { withAlpha } from "@/utils/color";

describe("withAlpha", () => {
    it("converts a 6-digit hex color to rgba with the given alpha", () => {
        expect(withAlpha("#9b59b6", 0.1)).toBe("rgba(155, 89, 182, 0.1)");
    });

    it("handles pure black", () => {
        expect(withAlpha("#000000", 0.5)).toBe("rgba(0, 0, 0, 0.5)");
    });

    it("handles pure white", () => {
        expect(withAlpha("#ffffff", 1)).toBe("rgba(255, 255, 255, 1)");
    });

    it("handles zero alpha", () => {
        expect(withAlpha("#3498db", 0)).toBe("rgba(52, 152, 219, 0)");
    });
});
