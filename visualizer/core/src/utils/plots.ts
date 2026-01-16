import Plotly from "plotly.js-dist-min";

export function disableAutoRangeFor(container: HTMLElement): void {
    Plotly.relayout(container, {
        "xaxis.autorange": false,
        "yaxis.autorange": false,
    });
}
