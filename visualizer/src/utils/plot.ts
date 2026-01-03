import type Plotly from "plotly.js-dist-min";

export function withoutAutorange(layout: Partial<Plotly.Layout>): Partial<Plotly.Layout> {
    return {
        ...layout,
        xaxis: {
            ...layout.xaxis,
            autorange: false,
        },
        yaxis: {
            ...layout.yaxis,
            autorange: false,
        },
    };
}
