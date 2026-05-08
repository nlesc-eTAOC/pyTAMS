"""A set of utility functions for database operations."""

from pathlib import Path
import numpy as np


def prepare_database_path(path: Path, restart: bool) -> None:
    """Archive a database if it already exists.

    Args:
        path: the path to the database
        restart: the database access mode
    """
    if path.exists() and restart:
        rng = np.random.default_rng(12345)
        while True:
            new_path = path.parent / f"{path.name}_{rng.integers(0, 999999):06d}"
            if not new_path.exists():
                break
        path.rename(new_path)
