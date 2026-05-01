from dataclasses import dataclass
from pytams.config import MergePolicy


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""

    __section__ = "database"
    __merge_policy__ = MergePolicy.IMMUTABLE

    path: str | None = None
    restart: bool = False
    format: str = "XML"
    archive_discarded: bool = True
