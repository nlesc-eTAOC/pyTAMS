from dataclasses import dataclass
from pytams.config import MergePolicy


@dataclass(frozen=True)
class TrajectoryConfig:
    """Trajectory configuration."""

    __section__ = "trajectory"
    __merge_policy__ = MergePolicy.IMMUTABLE

    step_size: float = -1.0
    end_time: float = -1.0
    targetscore: float = 0.95
    sparse_freq: int = 1
    sparse_start: int = 0
    chkfile_dump_all: bool = False

    def validate(self) -> None:
        """Validate trajectory configuration."""
        if self.step_size <= 0:
            err_msg = "TrajectoryConfig.step_size must be > 0"
            raise ValueError(err_msg)
