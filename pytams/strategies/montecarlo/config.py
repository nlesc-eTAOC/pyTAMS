from dataclasses import dataclass
from dataclasses import field
from pytams.core import MergePolicy


@dataclass(frozen=True)
class MCConfig:
    """MC strategy configuration."""

    __section__ = "montecarlo"
    __merge_policy__ = MergePolicy.IMMUTABLE

    ntrajectories: int = field(
        default=-1,
        metadata={
            "doc": "Number of trajectories to generate",
        },
    )

    end_time: float | None = field(
        default=None,
        metadata={
            "doc": "End time of the individual simulations",
        },
    )

    def validate(self) -> None:
        """Validate MC configuration."""
        if self.ntrajectories <= 0:
            err_msg = " MCConfig.ntrajectories must be > 0"
            raise ValueError(err_msg)
