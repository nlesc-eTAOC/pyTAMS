import dask
from dask.distributed import Client


class DaskRunner:
    """A Dask wrapper handle cluster and promises."""

    def __init__(self, n_daskTask : int =1):
        """Start the Dask cluster and client.

        Args:
            n_daskTask: number of dask workers
        """
        self.client = Client(threads_per_worker=1, n_workers=n_daskTask)

    def __enter__(self):
        """To enable use of with."""
        return self

    def __exit__(self, *args):
        """Executed leaving with."""
        self.client.close()

    def make_promise(self, task, *args):
        """Return a promise for a task."""
        return dask.delayed(task)(*args)

    def execute_promises(self, list_of_p : list):
        """Execute a list of promises.

        Args:
            list_of_p: a list of dask promises

        Return:
            A list with the return argument of each promised task.
        """
        return list(dask.compute(*list_of_p))
