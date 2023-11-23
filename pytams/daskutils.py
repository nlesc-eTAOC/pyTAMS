import dask
from dask.distributed import Client


class DaskRunner:
    def __init__(self, n_daskTask=1):
        self.client = Client(threads_per_worker=1, n_workers=n_daskTask)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def make_promise(self, task, *args):
        return dask.delayed(task)(*args)

    def execute_promises(self, list_of_p):
        return list(dask.compute(*list_of_p))
