"""A protocol database extension class for specific strategies."""

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

    def get_event_probability(self) -> float:
        """Return the event probability."""
        raise NotImplementedError
