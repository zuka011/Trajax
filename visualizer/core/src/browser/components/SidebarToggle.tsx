import type { FunctionalComponent } from "preact";
import { useCallback, useEffect, useState } from "preact/hooks";

export const SidebarToggle: FunctionalComponent = () => {
    const [isCollapsed, setIsCollapsed] = useState(false);

    const toggle = useCallback(() => {
        setIsCollapsed((prev) => !prev);
    }, []);

    useEffect(() => {
        const panel = document.getElementById("right-panel");
        if (panel) {
            if (isCollapsed) {
                panel.classList.add("collapsed");
            } else {
                panel.classList.remove("collapsed");
            }
        }
    }, [isCollapsed]);

    return (
        <button
            type="button"
            class={`sidebar-toggle ${isCollapsed ? "" : "active"}`}
            aria-label="Toggle sidebar"
            onClick={toggle}
        >
            <span class="sidebar-toggle-icon">{isCollapsed ? "≡" : "✕"}</span>
        </button>
    );
};
