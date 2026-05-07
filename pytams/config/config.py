from dataclasses import dataclass
from dataclasses import field
from .core import MergePolicy


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration."""

    __section__ = "runtime"
    __merge_policy__ = MergePolicy.REPLACE

    loglevel: str = field(
        default="INFO",
        metadata={
            "doc": "Logging level",
        },
    )

    logfile: str | None = field(
        default=None,
        metadata={
            "doc": "Logging file",
        },
    )

    walltime: float = field(
        default=86400,
        metadata={
            "doc": "Maximum walltime in seconds",
        },
    )

    plot_diagnostics: bool = field(
        default=False,
        metadata={
            "doc": "Diagnose ensemble by plotting scores",
        },
    )

    diagnostics: list[str] = field(
        default_factory=list,
        metadata={
            "doc": "List of diagnostics to compute",
        },
    )
