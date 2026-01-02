type VisibilityMap = Map<string, boolean>;

const globalVisibility: Map<string, VisibilityMap> = new Map();

function getPlotVisibility(plotId: string): VisibilityMap {
    if (!globalVisibility.has(plotId)) {
        globalVisibility.set(plotId, new Map());
    }
    return globalVisibility.get(plotId)!;
}

export function applyLegendState(traces: Plotly.Data[], plotId: string): Plotly.Data[] {
    const visibility = getPlotVisibility(plotId);

    return traces.map((trace) => {
        const key = ((trace as any).legendgroup as string) || (trace.name as string);
        if (!key) return trace;

        const isVisible = visibility.get(key);
        if (isVisible === undefined) return trace;

        return { ...trace, visible: isVisible ? true : "legendonly" };
    });
}
export function attachLegendHandler(
    container: HTMLElement,
    plotId: string,
    onToggle: () => void,
): void {
    const visibility = getPlotVisibility(plotId);

    (container as any).on?.("plotly_legendclick", (event: any) => {
        const trace = event.data[event.curveNumber];
        const key = trace.legendgroup || trace.name;
        if (!key) return true;

        const currentlyVisible = visibility.get(key) ?? true;

        if (trace.legendgroup) {
            event.data.forEach((t: any) => {
                if (t.legendgroup === trace.legendgroup) {
                    const k = t.legendgroup || t.name;
                    if (k) visibility.set(k, !currentlyVisible);
                }
            });
        } else {
            visibility.set(key, !currentlyVisible);
        }

        onToggle();
        return false;
    });
}
