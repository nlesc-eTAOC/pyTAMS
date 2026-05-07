from dataclasses import dataclass
from dataclasses import field
from pytams.config import MergePolicy


@dataclass(frozen=True)
class TrajectoryConfig:
    """Trajectory configuration."""

    __section__ = "trajectory"
    __merge_policy__ = MergePolicy.IMMUTABLE

    step_size: float = field(
        default=-1.0,
        metadata={
            "doc": "The stochastic time step size. Needs to be > 0.",
        },
    )
    end_time: float = field(
        default=-1.0,
        metadata={
            "doc": "The end time of the trajectory.",
        },
    )
    targetscore: float = field(
        default=0.95,
        metadata={
            "doc": "The target score for the trajectory.",
        },
    )
    sparse_freq: int = field(
        default=1,
        metadata={
            "doc": "The frequency at which the model state is stored in the trajectory.",
        },
    )
    sparse_start: int = field(
        default=0,
        metadata={
            "doc": "The first step at which the model state is stored in the trajectory.",
        },
    )
    chkfile_dump_all: bool = field(
        default=False,
        metadata={
            "doc": "Whether to dump all trajectory chkfile at every step.",
        },
    )

    def validate(self) -> None:
        """Validate trajectory configuration."""
        if self.step_size <= 0:
            err_msg = "TrajectoryConfig.step_size must be > 0"
            raise ValueError(err_msg)
