"""A base extension class for specific strategies."""

from typing import Protocol


class StrategyDatabaseExtension(Protocol):
    """A base extension class for specific strategies.

    Currently an empty type definition might be extended
    in the future for pickling purposes.
    """

    def serialize(self) -> None:
        """Serialize the extension."""
        raise NotImplementedError

    def deserialize(self) -> None:
        """Deserialize the extension."""
        raise NotImplementedError

    def get_rareevent_probability(self) -> float:
        """Return the rare-event probability."""
        raise NotImplementedError
