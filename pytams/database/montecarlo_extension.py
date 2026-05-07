"""An extension class for the MonteCarlo strategy."""

import logging
from typing import TypeVar
from .base_extension import StrategyDatabaseExtension
from .database import Database

_logger = logging.getLogger(__name__)

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class MCDatabaseExtension(StrategyDatabaseExtension):
    """An extension class for the MonteCarlo strategy."""

    def initialize(self, tdb: Database) -> None:
        """Initialize the AMS database extension.

        Args:
            nsplitting: maximum number of splitting iterations
            tdb: the core trajectory database
        """
        self._tdb = tdb

    def initialize_from_database(self, tdb: Database) -> None:
        """Initialize the AMS database extension.

        Args:
            tdb: the core trajectory database
        """

    def serialize(self) -> None:
        """Serialize the extension."""

    def deserialize(self) -> None:
        """Serialize the extension."""

    def get_rareevent_probability(self) -> float:
        """Return the rare-event probability."""
        return self._tdb.count_converged_traj() / self._tdb.n_traj()
