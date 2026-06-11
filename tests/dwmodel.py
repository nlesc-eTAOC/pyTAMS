import time
from typing import Any
import numpy as np
from pyrevs.core import ForwardModelBaseClass
from pyrevs.core import Snapshot


class DoubleWellModel(ForwardModelBaseClass):
    """2D double well forward model.

    V(x,y) = x^4/4 - x^2/2 + y^2

    Associated SDE:
    dX_t = -nabla V(X_t)dt + g(X_t)dW_t

    with:
    -nabla V(X_t) = [x - x^3, -2y]

    With the 2 wells at [-1.0, 0.0] and [1.0, 0.0]
    """

    def _init_model(self, m_id: int, params: dict[Any, Any] | None) -> None:
        """Override the template."""
        self._state = self.init_condition()
        if params is not None:
            self._slow_factor = params.get("slow_factor", 0.00000001)
            self._noise_amplitude = params.get("noise_amplitude", 1.0)
        if self._deterministic:
            self._rng = np.random.default_rng(m_id)
        else:
            self._rng = np.random.default_rng()

    def __RHS(self, state):
        """Double well RHS function."""
        sleepTime = float(self._slow_factor * np.random.rand(1).item())
        time.sleep(sleepTime)
        return np.array([state[0] - state[0] ** 3, -2 * state[1]])

    def __dW(self, dt, noise):
        """Stochastic forcing."""
        return np.sqrt(dt) * noise

    def init_condition(self):
        """Return the initial conditions."""
        return np.array([-1.0, 0.0])

    def _advance(self, step: int, time: float, dt: float, noise: Any, need_end_state: bool) -> float:
        """Override the template."""
        self._state = self._state + dt * self.__RHS(self._state) + self._noise_amplitude * self.__dW(dt, noise)
        return dt

    def get_current_state(self) -> Any:
        """Override the template."""
        return self._state

    def set_current_state(self, state: Any):
        """Override the template."""
        self._state = state

    def score(self):
        """Override the template."""
        a = np.array([-1.0, 0.0])
        b = np.array([1.0, 0.0])
        va = self._state - a
        vb = self._state - b
        da = np.sum(va**2, axis=0)
        db = np.sum(vb**2, axis=0)
        f1 = 0.5
        f2 = 1.0 - f1
        return f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)

    def make_noise(self):
        """Override the template."""
        return self._rng.standard_normal(2)

    def diagnostic_hook(
        self,
        dlabel: str,
        tid: int,
        score_level: float,
        old_snap: Snapshot,
        new_snap: Snapshot,
    ) -> None:
        """Override the template."""
        _, _, _, _ = dlabel, tid, score_level, old_snap
        old_factor = (score_level - old_snap.score) / (new_snap.score - old_snap.score)
        return old_factor * old_snap.state + (1 - old_factor) * new_snap.state

    @classmethod
    def name(cls) -> str:
        """Return the model name."""
        return "DoubleWellModel"
