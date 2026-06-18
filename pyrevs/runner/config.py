from dataclasses import dataclass
from dataclasses import field
from pyrevs.core import MergePolicy


@dataclass(frozen=True)
class DaskConfig:
    """Dask configuration."""

    __section__ = "dask"
    __merge_policy__ = MergePolicy.REPLACE

    backend: str = field(
        default="local",
        metadata={
            "doc": "The backend to use. Currently only `local` and `slurm` are supported.",
        },
    )
    slurm_config_file: str | None = field(
        default=None,
        metadata={
            "doc": "The path to the slurm config file.",
        },
    )
    one_worker_with_scheduler: bool = field(
        default=False,
        metadata={
            "doc": "Whether to assign one worker within the scheduler job.",
        },
    )
    queue: str = field(
        default="regular",
        metadata={
            "doc": "The slurm queue to use.",
        },
    )
    ntasks_per_job: int = field(
        default=1,
        metadata={
            "doc": "The number of tasks per job.",
        },
    )
    ntasks_per_node: int = field(
        default=-1,
        metadata={
            "doc": "The number of tasks per node.",
        },
    )
    ncores_per_worker: int = field(
        default=1,
        metadata={
            "doc": "The number of cores per worker.",
        },
    )
    job_prologue: list[str] = field(
        default_factory=list,
        metadata={
            "doc": "The job prologue commands, included before srun in dask slurm script.",
        },
    )
    worker_walltime: str = field(
        default="04:00:00",
        metadata={
            "doc": "The walltime for each worker, formatted as D:HH:MM:SS.",
        },
    )


@dataclass(frozen=True)
class RunnerConfig:
    """Runner configuration."""

    __section__ = "runner"
    __merge_policy__ = MergePolicy.REPLACE

    type: str = field(
        default="asyncio",
        metadata={
            "doc": "The type of runner to use. Currently only `asyncio` and `dask` are supported.",
        },
    )
    nworkers: int = field(
        default=1,
        metadata={
            "doc": "The number of workers.",
        },
    )
    dask_config: DaskConfig = field(
        default_factory=DaskConfig,
        metadata={
            "doc": "Dask configuration dictionary in runner.dask section.",
        },
    )
