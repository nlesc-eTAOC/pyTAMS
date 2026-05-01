from dataclasses import dataclass
from dataclasses import field
from .core import MergePolicy


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration."""

    __section__ = "runtime"
    __merge_policy__ = MergePolicy.REPLACE

    loglevel: str = "INFO"
    logfile: str | None = None
    walltime: float = 86400
    plot_diagnostics: bool = False
    diagnostics: list[str] = field(default_factory=list)
