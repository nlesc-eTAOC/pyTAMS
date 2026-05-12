"""Defines the interface and simple implementations of termination criteria."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar

if TYPE_CHECKING:
    from pytams.core import ForwardModelBaseClass
    from pytams.trajectory import Trajectory

T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class TerminationCriterion(Protocol):
    """Termination criterion interface."""

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> bool:
        """Check if the trajectory should terminate.

        Args:
            model: the forward model
            trajectory: the trajectory

        Returns:
            True if the trajectory should terminate
        """
        raise NotImplementedError


class TimeTerminationCriterion(TerminationCriterion):
    """Termination criterion based on time.

    Will trigger termination if the current time is greater than or equal
    to the end time.
    """

    def __init__(self, end_time: float) -> None:
        """Initialize the termination criterion.

        Args:
            end_time: the end time
        """
        self._end_time = end_time

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> bool:
        """Check if the trajectory should terminate."""
        _ = model
        return trajectory.current_time() >= self._end_time


class LowScoreTerminationCriterion(TerminationCriterion):
    """Termination criterion based on score.

    Will trigger termination if the current score is less than or equal
    to a threshold.

    """

    def __init__(self, score_threshold: float) -> None:
        """Initialize the termination criterion.

        Args:
            score_threshold: the score threshold
        """
        self._score_threshold = score_threshold

    def should_terminate(
        self,
        model: ForwardModelBaseClass[T_Noise, T_State],
        trajectory: Trajectory,
    ) -> bool:
        """Check if the trajectory should terminate."""
        _ = trajectory
        return model.score() <= self._score_threshold
