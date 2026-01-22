import { actualPathUpdater } from "./actual";
import { boundariesUpdater } from "./boundary";
import { forecastsUpdater } from "./forecasts";
import { ghostUpdater } from "./ghost";
import { obstaclesUpdater } from "./obstacles";
import { referencePathUpdater } from "./reference";
import { roadNetworkUpdater } from "./road";
import { nominalTrajectoryUpdater, optimalTrajectoryUpdater } from "./trajectories";
import { uncertaintiesUpdater } from "./uncertainties";
import type { TraceUpdateCreator } from "./updater";
import { vehicleUpdater } from "./vehicle";

export namespace updaterCreator {
    export const roadNetwork: TraceUpdateCreator = roadNetworkUpdater;
    export const referencePath: TraceUpdateCreator = referencePathUpdater;
    export const boundaries: TraceUpdateCreator = boundariesUpdater;
    export const actualPath: TraceUpdateCreator = actualPathUpdater;
    export const vehicle: TraceUpdateCreator = vehicleUpdater;
    export const ghost: TraceUpdateCreator = ghostUpdater;
    export const optimalTrajectory: TraceUpdateCreator = optimalTrajectoryUpdater;
    export const nominalTrajectory: TraceUpdateCreator = nominalTrajectoryUpdater;
    export const obstacles: TraceUpdateCreator = obstaclesUpdater;
    export const obstacleForecasts: TraceUpdateCreator = forecastsUpdater;
    export const forecastUncertainties: TraceUpdateCreator = uncertaintiesUpdater;
}
