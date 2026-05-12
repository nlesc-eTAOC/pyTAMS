from __future__ import annotations
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from .database import Database

if TYPE_CHECKING:
    from pathlib import Path


def load_database(
    path: Path,
    read_only: bool = True,
) -> Database:
    """Load a database with its strategy extension."""
    db: Database = Database.load(path, read_only=read_only)

    strategy = db.strategy()

    eps = entry_points(group="pyrevs.strategies")

    try:
        ep = next(ep for ep in eps if ep.name == strategy)
    except StopIteration as exc:
        err_msg = f"No plugin found for strategy '{strategy}'"
        raise RuntimeError(err_msg) from exc

    plugin = ep.load()

    extension = plugin.load_database_extension(db)

    db.attach_extension(extension)

    return db
