import type { FunctionalComponent } from "preact";
import type { ProcessedSimulationData } from "../../core/types.js";
import { radiansToDegrees } from "../../utils/math.js";

interface InfoPanelProps {
    currentTimestep: number;
    data: ProcessedSimulationData;
}

export const InfoPanel: FunctionalComponent<InfoPanelProps> = ({ currentTimestep, data }) => {
    const t = currentTimestep;
    const time = (t * data.info.timeStep).toFixed(2);
    const posX = data.ego.x[t].toFixed(2);
    const posY = data.ego.y[t].toFixed(2);
    const heading = radiansToDegrees(data.ego.heading[t]).toFixed(1);
    const pathParam = data.ego.pathParameter[t].toFixed(2);
    const pathLength = data.info.pathLength.toFixed(1);
    const progress = ((100 * data.ego.pathParameter[t]) / data.info.pathLength).toFixed(1);

    return (
        <div class="info-panel" id="info-panel">
            <h3>Simulation State</h3>
            <table class="info-table">
                <tr>
                    <td>Time</td>
                    <td>{time} s</td>
                </tr>
                <tr>
                    <td>Position</td>
                    <td>
                        ({posX}, {posY}) m
                    </td>
                </tr>
                <tr>
                    <td>Heading</td>
                    <td>{heading}°</td>
                </tr>
                <tr>
                    <td>Path Parameter</td>
                    <td>
                        {pathParam} / {pathLength} m
                    </td>
                </tr>
                <tr>
                    <td>Progress</td>
                    <td>{progress}%</td>
                </tr>
            </table>
        </div>
    );
};
