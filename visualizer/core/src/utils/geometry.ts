const BOX_POINT_COUNT = 5;
const ELLIPSE_SEGMENTS = 36;
const ELLIPSE_POINT_COUNT = ELLIPSE_SEGMENTS + 1;
const HEADING_LINE_POINT_COUNT = 2;

const CORNERS_X = [-0.5, 0.5, 0.5, -0.5, -0.5] as const;
const CORNERS_Y = [-0.5, -0.5, 0.5, 0.5, -0.5] as const;

const [UNIT_COS, UNIT_SIN] = generateUnitCirclePoints();

function generateUnitCirclePoints(): [Float64Array, Float64Array] {
    const cosArray = new Float64Array(ELLIPSE_POINT_COUNT);
    const sinArray = new Float64Array(ELLIPSE_POINT_COUNT);

    for (let i = 0; i < ELLIPSE_POINT_COUNT; i++) {
        const angle = (i / ELLIPSE_SEGMENTS) * 2 * Math.PI;
        cosArray[i] = Math.cos(angle);
        sinArray[i] = Math.sin(angle);
    }

    return [cosArray, sinArray];
}

export const write = {
    box: Object.assign(
        (
            cx: number,
            cy: number,
            heading: number,
            length: number,
            width: number,
            out: { x: (number | null)[]; y: (number | null)[] },
            offset: number,
        ): number => {
            const cos = Math.cos(heading);
            const sin = Math.sin(heading);

            for (let i = 0; i < BOX_POINT_COUNT; i++) {
                const dx = CORNERS_X[i] * length;
                const dy = CORNERS_Y[i] * width;
                out.x[offset + i] = cx + dx * cos - dy * sin;
                out.y[offset + i] = cy + dx * sin + dy * cos;
            }

            return offset + BOX_POINT_COUNT;
        },
        { pointCount: BOX_POINT_COUNT },
    ),

    ellipse: Object.assign(
        (
            cx: number,
            cy: number,
            rx: number,
            ry: number,
            angle: number,
            out: { x: (number | null)[]; y: (number | null)[] },
            offset: number,
        ): number => {
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);

            for (let i = 0; i < ELLIPSE_POINT_COUNT; i++) {
                const ex = rx * UNIT_COS[i];
                const ey = ry * UNIT_SIN[i];
                out.x[offset + i] = cx + ex * cos - ey * sin;
                out.y[offset + i] = cy + ex * sin + ey * cos;
            }

            return offset + ELLIPSE_POINT_COUNT;
        },
        { pointCount: ELLIPSE_POINT_COUNT },
    ),

    headingLine: Object.assign(
        (
            cx: number,
            cy: number,
            radius: number,
            heading: number,
            out: { x: (number | null)[]; y: (number | null)[] },
            offset: number,
        ): number => {
            const cos = Math.cos(heading);
            const sin = Math.sin(heading);
            out.x[offset] = cx;
            out.y[offset] = cy;
            out.x[offset + 1] = cx + radius * cos;
            out.y[offset + 1] = cy + radius * sin;

            return offset + HEADING_LINE_POINT_COUNT;
        },
        { pointCount: HEADING_LINE_POINT_COUNT },
    ),
};
