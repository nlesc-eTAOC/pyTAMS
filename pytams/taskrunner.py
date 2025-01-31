from __future__ import annotations
import asyncio
import concurrent.futures
import ntpath
import os
import shutil
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Callable
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


class RunnerError(Exception):
    """Exception class for the runner."""

    pass

class BaseRunner(metaclass=ABCMeta):
    """An ABC for the task runners."""

    @abstractmethod
    def __init__(self,
                 params: dict,
                 sync_wk : Callable,
                 async_wk : Callable,
                 n_workers: int = 1,
                 ):
        """A dummy init method."""
        pass

    @abstractmethod
    def __enter__(self) -> BaseRunner:
        """To enable use of with."""
        pass

    @abstractmethod
    def __exit__(self, *args : list[str]) -> None:
        """Executed leaving with scope."""
        pass

    @abstractmethod
    def make_promise(self, task : list[Any]) -> None:
        """Log a new task to the list of task to tackle."""
        pass

    @abstractmethod
    def execute_promises(self) -> Any:
        """Execute the list of promises."""
        pass

    @abstractmethod
    def n_workers(self) -> int:
        """Return the number of workers in the runner."""
        pass


class AsIORunner(BaseRunner):
    """A task runner class based on asyncIO.

    An runner that relies on asyncio to schedule
    a tasks concurently in worker processes.
    Tasks are added to an internal queue from
    which worker can take them and put the results
    back into result queue.
    """

    def __init__(self,
                 params: dict,
                 sync_wk : Callable,
                 async_wk : Callable,
                 n_workers: int = 1,
                 ):
        """Init the task runner.

        Args:
            params: a dictionary of parameters
            sync_wk: a synchronous worker function
            async_wk: an asynchronous worker function
            n_workers: number of workers
        """
        self._queue : asyncio.Queue[Any] = asyncio.Queue()
        self._rqueue : asyncio.Queue[Any] = asyncio.Queue()
        self._n_workers : int = n_workers
        self._sync_worker = sync_wk
        self._async_worker = async_wk
        self._loop : asyncio.AbstractEventLoop | None = None
        self._executor : concurrent.futures.Executor | None = None
        self._workers : list[asyncio.Task[Any]] | None = None

    def __enter__(self) -> AsIORunner:
        """To enable use of with."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self

    def __exit__(self, *args : list[str]) -> None:
        """Executed leaving with scope."""
        if self._workers:
            for w in self._workers:
                w.cancel()
        if self._executor:
            self._executor.shutdown()
        if self._loop:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
            asyncio.set_event_loop(None)

    async def add_task(self, task : list[Any]) -> None:
        """Append a task to the queue."""
        await self._queue.put(task)

    def make_promise(self, task : list[Any]) -> None:
        """A synchronous wrapper to add_task."""
        asyncio.run(self.add_task(task))

    async def run_tasks(self) -> list[Any]:
        """Create worker tasks and run."""
        if not self._workers:
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self._n_workers)
            self._workers = [
                asyncio.create_task(self._async_worker(self._queue,
                                                       self._rqueue,
                                                       self._executor))
                for _ in range(self._n_workers)]

        # Wait until all tasks are processed
        await self._queue.join()

        res = []
        while not self._rqueue.empty():
            res.append(await self._rqueue.get())

        return res

    def execute_promises(self) -> Any:
        """A synchronous wrapper to run_tasks."""
        assert(self._loop)
        return self._loop.run_until_complete(self.run_tasks())

    def n_workers(self) -> int:
        """Return the number of workers in the runner."""
        return self._n_workers

class DaskRunner(BaseRunner):
    """A task runner class based on Dask.

    An runner that relies on dask to schedule
    a tasks concurently in workers.
    """

    def __init__(self,
                 params: dict,
                 sync_wk : Callable,
                 async_wk : Callable,
                 n_workers: int = 1,
                 ):
        """Start the Dask cluster and client.

        Args:
            params: a dictionary with params
            sync_wk: a synchronous worker function
            async_wk: an asynchronous worker function
            n_workers: number of workers
        """
        dask_dict = params.get("dask",{})
        self.dask_backend = dask_dict.get("backend", "local")
        self._n_workers : int = n_workers
        self._sync_worker = sync_wk
        self._async_worker = async_wk
        self._tlist : list[Any] = []
        if self.dask_backend == "local":
            self.client = Client(threads_per_worker=1, n_workers=self._n_workers)
            self.cluster = None
        elif self.dask_backend == "slurm":
            self.slurm_config_file = dask_dict.get("slurm_config_file", None)
            if self.slurm_config_file:
                if not os.path.exists(self.slurm_config_file):
                    raise RunnerError(
                        "Specified slurm_config_file do not exists: {}".format(
                            self.slurm_config_file
                        )
                    )

                config_file = ntpath.basename(self.slurm_config_file)
                shutil.move(
                    self.slurm_config_file, "~/.config/dask/{}".format(config_file)
                )
                self.cluster = SLURMCluster()
            else:
                self.dask_queue = dask_dict.get("queue", "regular")
                self.dask_ntasks = dask_dict.get("ntasks_per_job", 1)
                self.dask_ntasks_per_node = dask_dict.get("ntasks_per_node", self.dask_ntasks)
                self.dask_nworker_ncore = dask_dict.get("ncores_per_worker", 1)
                self.dask_prologue = dask_dict.get("job_prologue", [])
                self.dask_walltime = dask_dict.get("worker_walltime", "04:00:00")
                self.cluster = SLURMCluster(
                    queue=self.dask_queue,
                    cores=self.dask_nworker_ncore,
                    memory='144GB',
                    walltime=self.dask_walltime,
                    processes=1,
                    interface='ib0',
                    job_script_prologue=self.dask_prologue,
                    job_extra_directives=["--ntasks={}".format(self.dask_ntasks),
                                          "--tasks-per-node={}".format(self.dask_ntasks_per_node),
                                          "--exclusive"],
                    job_directives_skip=['--cpus-per-task=', '--mem'],
                )
            self.cluster.scale(jobs=self._n_workers)
            self.client = Client(self.cluster)
        else:
            raise RunnerError("Unknown [dask] backend: {}".format(self.dask_backend))

    def __enter__(self) -> DaskRunner:
        """To enable use of with."""
        return self

    def __exit__(self, *args : list[str]) -> None:
        """Executed leaving with scope."""
        if self.cluster:
            self.cluster.close()
        self.client.close()

    def make_promise(self, task: list[Any]) -> None:
        """Append a task to the internal task list."""
        self._tlist.append(dask.delayed(self._sync_worker)(*task))

    def just_delay(self, obj : Any) -> Any:
        """Delay an object."""
        return dask.delayed(obj)

    def execute_promises(self) -> Any:
        """Execute a list of promises.

        Args:
            list_of_p: a list of dask promises

        Return:
            A list with the return argument of each promised task.
        """
        res = list(dask.compute(*self._tlist))
        self._tlist.clear()
        return res

    def n_workers(self) -> int:
        """Return the number of workers in the runner."""
        return self._n_workers


def get_runner_type(params : dict) -> type[BaseRunner]:
    """Create an engine from parameters."""
    runner_map = {
        "dask": DaskRunner,
        "asyncio": AsIORunner,
    }
    runner_str = params.get("runner", {}).get("type").lower()
    if runner_str not in runner_map:
        raise RunnerError("Unable to get unknown {} runner.".format(runner_str))

    return runner_map[runner_str]
