"""An extension class for the GKTL strategy."""

import gc
import json
import logging
from pathlib import Path
from typing import TypeVar
from pyrevs.database import Database
from pyrevs.database import StrategyDatabaseExtension

_logger = logging.getLogger(__name__)

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class GKTLDatabaseExtension(StrategyDatabaseExtension):
    """An extension class for the GKTL strategy.

    An extension of the database to store GKTL-specific
    data. To be able to checkpoint a sampling run with GKTL,
    one need to store/read the following attributes:

    Attributes:
        _iteration_count: number of iterations
        _cur_time: current time
    """

    def __init__(self) -> None:
        self._iteration_count = 0
        self._cur_time = 0.0

    def initialize(self, tdb: Database) -> None:
        """Initialize the AMS database extension.

        Args:
            tdb: the core trajectory database
        """
        self._tdb = tdb

    def initialize_from_database(self, tdb: Database) -> None:
        """Initialize the AMS database extension.

        Args:
            tdb: the core trajectory database
        """
        self._tdb = tdb
        self.deserialize()

    def serialize(self) -> None:
        """Serialize the extension."""
        spath = Path(self._tdb.name()) / "gktl_metadata.json"
        data = {
            "iteration_count": self._iteration_count,
            "cur_time": self._cur_time,
        }
        with spath.open("w") as f:
            json.dump(data, f, indent=2)

    def deserialize(self) -> None:
        """Serialize the extension."""
        spath = Path(self._tdb.name()) / "gktl_metadata.json"
        with spath.open("r") as f:
            data = json.load(f)
        self._iteration_count = data["iteration_count"]
        self._cur_time = data["cur_time"]

    def get_event_probability(self) -> float:
        """Return the event probability."""
        return 0.0

    def __del__(self) -> None:
        """Destructor of the GKTL extension.

        Force deletion of SQL accesses for Windows.
        """
        if hasattr(self, "_tdb"):
            del self._tdb
        gc.collect()
