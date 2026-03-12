from __future__ import annotations
from dataclasses import dataclass
from typing import Generic
from typing import TypeVar

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


@dataclass(frozen=True, slots=True, kw_only=True)
class Snapshot(Generic[T_Noise, T_State]):
    """A dataclass defining a snapshot.

    Gathering what defines a snapshot into an object.
    The time and score are of float type, but the
    actual type of the noise and state are completely
    determined by the forward model.
    A snapshot is allowed to have a state or not to
    accommodate memory savings.

    Attributes:
        time : snapshot time
        score : score function value
        noise : noise used to reach this snapshot
        state : model state
    """

    time: float
    score: float
    noise: T_Noise
    state: T_State | None = None

    def __post_init__(self) -> None:
        """Check that the initial fields are sensible."""
        if self.time < 0.0:
            err_msg = f"Time cannot be negative (input {self.time}"
            raise ValueError(err_msg)

    @property
    def has_state(self) -> bool:
        """Check if snapshot has state.

        Returns:
            bool : True if state is not None
        """
        return self.state is not None
