import ntpath
import os
import shutil
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


class DaskRunnerError(Exception):
    """Exception class for the Dask runner."""

    pass


class DaskRunner:
    """A Dask wrapper handle cluster and promises."""

    def __init__(self, parameters: dict, is_init: bool = True):
        """Start the Dask cluster and client.

        Args:
            parameters: a dictionary with parameters
        """
        self.dask_backend = parameters.get("dask.backend", "local")
        if self.dask_backend == "local":
            if (is_init):
                self.dask_nworker = parameters.get("dask.nworker_init", 1)
            else:
                self.dask_nworker = parameters.get("dask.nworker_iter", 1)
            self.client = Client(threads_per_worker=1, n_workers=self.dask_nworker)
            self.cluster = None
        elif self.dask_backend == "slurm":
            if (is_init):
                self.dask_nworker = parameters.get("dask.nworker_init", 1)
            else:
                self.dask_nworker = parameters.get("dask.nworker_iter", 1)
            self.slurm_config_file = parameters.get("dask.slurm_config_file", None)
            if self.slurm_config_file:
                if not os.path.exists(self.slurm_config_file):
                    raise DaskRunnerError(
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
                self.dask_queue = parameters.get("dask.queue", "regular")
                self.dask_nworker_ncore = parameters.get("dask.ncores_per_worker", 1)
                self.cluster = SLURMCluster(
                    queue=self.dask_queue,
                    cores=self.dask_nworker_ncore,
                    walltime="00:30:00",
                )
            self.cluster.scale(jobs=self.dask_nworker)
            self.client = Client(self.cluster)
        else:
            raise DaskRunnerError("Unknown dask.backend: {}".format(self.dask_backend))

    def __enter__(self):
        """To enable use of with."""
        return self

    def __exit__(self, *args):
        """Executed leaving with scope."""
        if self.cluster:
            self.cluster.close()
        self.client.close()

    def make_promise(self, task, *args):
        """Return a promise for a task."""
        return dask.delayed(task)(*args)

    def just_delay(self, obj):
        """Delay an object."""
        return dask.delayed(obj)

    def execute_promises(self, list_of_p: list):
        """Execute a list of promises.

        Args:
            list_of_p: a list of dask promises

        Return:
            A list with the return argument of each promised task.
        """
        return list(dask.compute(*list_of_p))
