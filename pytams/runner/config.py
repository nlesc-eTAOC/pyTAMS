from dataclasses import dataclass
from dataclasses import field
from pytams.config import MergePolicy


@dataclass(frozen=True)
class DaskConfig:
    """Dask configuration."""

    __section__ = "dask"
    __merge_policy__ = MergePolicy.REPLACE

    backend: str = "local"
    slurm_config_file: str | None = None
    queue: str = "regular"
    ntasks_per_job: int = 1
    ntasks_per_node: int = ntasks_per_job
    ncores_per_worker: int = 1
    job_prologue: list[str] = field(default_factory=list)
    worker_walltime: str = "04:00:00"


@dataclass(frozen=True)
class RunnerConfig:
    """Runner configuration."""

    __section__ = "runner"
    __merge_policy__ = MergePolicy.REPLACE

    type: str = "asyncio"
    nworkers_init: int = 1
    nworkers_iter: int = 1
    dask_config: DaskConfig = field(default_factory=DaskConfig)
