from pyrevs.database import Database
from .ams import AMS
from .config import AMSConfig
from .extension import AMSDatabaseExtension

__all__ = ["AMS", "AMSConfig", "AMSDatabaseExtension", "load_database_extension"]


def load_database_extension(tdb: Database) -> AMSDatabaseExtension:
    """A factory function to instanciate the extension from the database."""
    ext = AMSDatabaseExtension()
    ext.initialize_from_database(tdb)
    return ext
