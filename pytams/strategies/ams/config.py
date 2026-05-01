from dataclasses import dataclass
from pytams.config import MergePolicy


@dataclass(frozen=True)
class AMSConfig:
    """AMS strategy configuration."""

    __section__ = "ams"
    __merge_policy__ = MergePolicy.IMMUTABLE

    ntrajectories: int = -1
    nsplititer: int = -1
    variant: str = "ams"
    init_ensemble_only: bool = False

    def validate(self) -> None:
        """Validate AMS configuration."""
        if self.ntrajectories <= 0:
            err_msg = " AMSConfig.ntrajectories must be > 0"
            raise ValueError(err_msg)

        if self.nsplititer <= 0:
            err_msg = " AMSConfig.nsplititer must be > 0"
            raise ValueError(err_msg)
