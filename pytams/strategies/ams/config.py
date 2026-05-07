from dataclasses import dataclass
from dataclasses import field
from pytams.config import MergePolicy


@dataclass(frozen=True)
class AMSConfig:
    """AMS strategy configuration."""

    __section__ = "ams"
    __merge_policy__ = MergePolicy.IMMUTABLE

    ntrajectories: int = field(
        default=-1,
        metadata={
            "doc": "Number of trajectories to sample. Sensible value checked upon initialization.",
        },
    )
    nsplititer: int = field(
        default=-1,
        metadata={
            "doc": "Number of splitting iterations. Sensible value checked upon initialization.",
        },
    )
    variant: str = field(
        default="tams",
        metadata={
            "doc": "Variant of AMS to use (one of [tams, ams])",
        },
    )
    init_ensemble_only: bool = field(
        default=False,
        metadata={
            "doc": "Whether or not to stop after initializing the trajectory ensemble",
        },
    )

    def validate(self) -> None:
        """Validate AMS configuration."""
        if self.ntrajectories <= 0:
            err_msg = " AMSConfig.ntrajectories must be > 0"
            raise ValueError(err_msg)

        if self.nsplititer <= 0:
            err_msg = " AMSConfig.nsplititer must be > 0"
            raise ValueError(err_msg)
