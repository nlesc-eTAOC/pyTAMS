from pytams.database import Database
from .config import MCConfig
from .extension import MCDatabaseExtension
from .montecarlo import MonteCarlo

__all__ = ["MCConfig", "MCDatabaseExtension", "MonteCarlo", "load_database_extension"]


def load_database_extension(tdb: Database) -> MCDatabaseExtension:
    """A factory function to instanciate the extension from the database."""
    _ = tdb
    return MCDatabaseExtension()
