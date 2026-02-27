import time
from typing import Any
import numpy as np
from pytams.fmodel import ForwardModelBaseClass


class SimpleFModel(ForwardModelBaseClass):
    """Simple forward model.

    The state is the time and score
    10 times the state, ceiled to 1.0
    """

    def _init_model(self, m_id: int, params: dict[Any, Any]) -> None:
        """Initialize model state."""
        _, _ = m_id, params
        self._state: float = 0.0

    def _advance(self, step: int, time: float, dt: float, noise: Any, need_end_state: bool) -> float:
        """Override the template."""
        self._state = self._state + dt
        return dt

    def get_current_state(self) -> float:
        """Override the template."""
        return self._state

    def set_current_state(self, state) -> None:
        """Override the template."""
        self._state = state

    def score(self) -> float:
        """Override the template."""
        return min(self._state * 10.0, 1.0)

    def make_noise(self) -> float:
        """Override the template."""
        return 0.0

    @classmethod
    def name(cls) -> str:
        """Return the model name."""
        return "SimpleFModel"


class FailingFModel(ForwardModelBaseClass):
    """Simple failing forward model.

    The state is the time and score
    10 times the state, ceiled to 1.0
    The model thow an exception if the score exceed 0.5
    """

    def _init_model(self, m_id: int, params: dict[Any, Any]) -> None:
        """Initialize model state."""
        _, _ = m_id, params
        self._state: float = 0.0

    def _advance(self, step: int, time: float, dt: float, noise: Any, need_end_state: bool) -> float:
        """Override the template."""
        if self.score() > 0.5:
            raise RuntimeError("Failing model")
        self._state = self._state + dt
        return dt

    def get_current_state(self) -> float:
        """Override the template."""
        return self._state

    def set_current_state(self, state) -> None:
        """Override the template."""
        self._state = state

    def score(self) -> float:
        """Override the template."""
        return min(self._state * 10.0, 1.0)

    def make_noise(self) -> float:
        """Override the template."""
        return 0.0

    @classmethod
    def name(cls) -> str:
        """Return the model name."""
        return "SimpleFModel"
