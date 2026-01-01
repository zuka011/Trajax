import type { FunctionalComponent } from "preact";
import type { ProcessedSimulationData } from "../../core/types.js";
import { radiansToDegrees } from "../../utils/math.js";

interface InfoPanelProps {
    currentTimestep: number;
    data: ProcessedSimulationData;
}

export const InfoPanel: FunctionalComponent<InfoPanelProps> = ({ currentTimestep, data }) => {
    const t = currentTimestep;
    const time = (t * data.time_step).toFixed(2);
    const posX = data.positions_x[t].toFixed(2);
    const posY = data.positions_y[t].toFixed(2);
    const heading = radiansToDegrees(data.headings[t]).toFixed(1);
    const pathParam = data.path_parameters[t].toFixed(2);
    const pathLength = data.path_length.toFixed(1);
    const progress = ((100 * data.path_parameters[t]) / data.path_length).toFixed(1);

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
