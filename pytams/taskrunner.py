import asyncio
import concurrent.futures
from typing import Any, Optional

class TaskRunner:
    """A task runner class.

    """

    def __init__(self,
                 params: dict,
                 worker,
                 n_workers: int = 1):
        """Init the task runner."""
        self._queue = asyncio.Queue()
        self._rqueue = asyncio.Queue()
        self._n_workers : int = n_workers
        self._worker = worker
        self._loop = None

    def __enter__(self):
        """To enable use of with."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        print(id(self._loop))
        print(id(asyncio.get_event_loop()))
        return self

    def __exit__(self, *args : list[str]) -> None:
        """Executed leaving with scope."""
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()
        asyncio.set_event_loop(None)

    async def add_task(self, task):
        """Append a task to the queue."""
        await self._queue.put(task)

    def make_promise(self, task):
        asyncio.run(self.add_task(task))

    async def run_tasks(self, cancel_workers : Optional[bool] = True):
        """Create worker tasks and run."""

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            workers = [asyncio.create_task(self._worker(self._queue, self._rqueue, executor)) for i in range(self._n_workers)]

            # Wait until all tasks are processed
            await self._queue.join()

            # Cancel worker tasks
            if cancel_workers:
                for w in workers:
                    w.cancel()

            # Wait until all worker tasks are cancelled
            await asyncio.gather(*workers, return_exceptions=True)

        res = []
        while not self._rqueue.empty():
            res.append(await self._rqueue.get())

        return res

    def execute_promises(self, cancel_workers : Optional[bool] = True) -> Any:
        return self._loop.run_until_complete(self.run_tasks(cancel_workers))

    def n_workers(self) -> int:
        return self._n_workers
