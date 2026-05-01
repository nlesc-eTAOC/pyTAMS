from dataclasses import dataclass
from pytams.config import MergePolicy


@dataclass(frozen=True)
class MCConfig:
    """MC strategy configuration."""

    __section__ = "montecarlo"
    __merge_policy__ = MergePolicy.IMMUTABLE

    ntrajectories: int = -1

    def validate(self) -> None:
        """Validate MC configuration."""
        if self.ntrajectories <= 0:
            err_msg = " MCConfig.ntrajectories must be > 0"
            raise ValueError(err_msg)
