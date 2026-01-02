import asyncio
import threading
import traceback
from typing import Self, Coroutine
from types import TracebackType
from dataclasses import dataclass, field


@dataclass
class BackgroundTasks:
    _loop: asyncio.AbstractEventLoop = field(init=False)
    _thread: threading.Thread = field(init=False)
    _pending: list[asyncio.Future] = field(default_factory=list)

    async def __aenter__(self) -> Self:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            print(f"BackgroundTasks exiting with exception: {exc_val!r}")
            traceback.print_exception(exc_type, exc_val, exc_tb)

        await self._wait()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def schedule(self, task: Coroutine) -> None:
        future = asyncio.run_coroutine_threadsafe(task, self._loop)
        self._pending.append(future)  # type: ignore

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _wait(self) -> None:
        if not self._pending:
            return

        print(f"\nWaiting for {len(self._pending)} background task(s)...")

        await asyncio.gather(*[asyncio.wrap_future(f) for f in self._pending])

        print("All background tasks completed.\n")

        self._pending.clear()
