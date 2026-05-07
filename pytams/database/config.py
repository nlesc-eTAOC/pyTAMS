from dataclasses import dataclass
from dataclasses import field
from pytams.config import MergePolicy


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""

    __section__ = "database"
    __merge_policy__ = MergePolicy.IMMUTABLE

    path: str | None = field(
        default=None,
        metadata={
            "doc": "path to the database folder (DB not saved if None)",
        },
    )

    restart: bool = field(
        default=False,
        metadata={
            "doc": "force restart the database: pre-existing database is archived",
        },
    )

    format: str = field(
        default="XML",
        metadata={
            "doc": "database format (only XML supported for now)",
        },
    )

    archive_discarded: bool = field(
        default=True,
        metadata={
            "doc": "archive discarded trajectories",
        },
    )
