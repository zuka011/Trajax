import type { FunctionalComponent } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import type { Plot, Visualizable } from "../../core/types.js";
import type { VisualizationState } from "../state.js";
import type { UpdateManager } from "../update.js";
import { createAdditionalPlot, groupPlots, type PlotGroup } from "./additional-plot.js";

interface PlotContainerProps {
    data: Visualizable.ProcessedSimulationResult;
    state: VisualizationState;
    updateManager: UpdateManager;
}

interface TabHeaderProps {
    groups: PlotGroup[];
    activeIndex: number;
    onTabChange: (index: number) => void;
}

const TabHeader: FunctionalComponent<TabHeaderProps> = ({ groups, activeIndex, onTabChange }) => (
    <div class="plot-tabs-header">
        {groups.map((group, index) => (
            <button
                key={group.id}
                type={"button"}
                class={`plot-tab ${index === activeIndex ? "active" : ""}`}
                onClick={() => onTabChange(index)}
                title={group.name}
            >
                {truncateName(group.name, 15)}
            </button>
        ))}
    </div>
);

function truncateName(name: string, maxLength: number): string {
    return name.length <= maxLength ? name : `${name.slice(0, maxLength - 1)}…`;
}

interface SinglePlotPanelProps {
    group: PlotGroup;
    data: Visualizable.ProcessedSimulationResult;
    state: VisualizationState;
    updateManager: UpdateManager;
}

const SinglePlotPanel: FunctionalComponent<SinglePlotPanelProps> = ({
    group,
    data,
    state,
    updateManager,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const initializedRef = useRef(false);

    useEffect(() => {
        if (containerRef.current && !initializedRef.current) {
            initializedRef.current = true;
            requestAnimationFrame(() => {
                if (containerRef.current) {
                    createAdditionalPlot(containerRef.current, group, data, state, updateManager);
                }
            });
        }
    }, [group.id]);

    return <div ref={containerRef} class="additional-plot-content" />;
};

export const AdditionalPlotsContainer: FunctionalComponent<PlotContainerProps> = ({
    data,
    state,
    updateManager,
}) => {
    const plots: Plot.Additional[] = data.additionalPlots ?? [];

    if (plots.length === 0) {
        return null;
    }

    const groups = groupPlots(plots);
    const [activeTabIndex, setActiveTabIndex] = useState(0);
    const showTabs = groups.length > 2;
    const visibleGroups = showTabs ? [groups[activeTabIndex]] : groups;

    return (
        <div class="additional-plots-container">
            {showTabs && (
                <TabHeader
                    groups={groups}
                    activeIndex={activeTabIndex}
                    onTabChange={setActiveTabIndex}
                />
            )}
            <div class={`additional-plots-content ${showTabs ? "tabbed" : ""}`}>
                {visibleGroups.map((group) => (
                    <div key={group.id} class="plot-container">
                        <SinglePlotPanel
                            group={group}
                            data={data}
                            state={state}
                            updateManager={updateManager}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
};
