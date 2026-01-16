import { bandsUpdater } from "./bands";
import { boundsUpdater } from "./bounds";
import { seriesLinesUpdater } from "./seriesLines";
import { seriesMarkersUpdater } from "./seriesMarkers";
import type { AdditionalPlotUpdaterCreator } from "./updater";

export namespace updaterCreator {
    export const seriesLines: AdditionalPlotUpdaterCreator = seriesLinesUpdater;
    export const seriesMarkers: AdditionalPlotUpdaterCreator = seriesMarkersUpdater;
    export const bands: AdditionalPlotUpdaterCreator = bandsUpdater;
    export const bounds: AdditionalPlotUpdaterCreator = boundsUpdater;
}
