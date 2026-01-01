export interface UpdateManager {
    /**
     * Notify all subscribers of an update.
     */
    notify(): void;

    /**
     * Subscribe to updates.
     *
     * @param callback - The callback to invoke on updates.
     */
    subscribe(callback: () => void): void;
}

export function createUpdateManager(): UpdateManager {
    const subscribers: (() => void)[] = [];

    return {
        notify(): void {
            for (const callback of subscribers) {
                callback();
            }
        },
        subscribe(callback: () => void): void {
            subscribers.push(callback);
        },
    };
}
