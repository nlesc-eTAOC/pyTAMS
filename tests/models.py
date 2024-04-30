import time
from typing import Optional
import numpy as np
from pytams.fmodel import ForwardModel


class SimpleFModel(ForwardModel):
    """Simple forward model.

    The state is the time and score
    10 times the state, ceiled to 1.0
    """

    def __init__(self,
                 params: Optional[dict] = None,
                 ioprefix: Optional[str] = None):
        """Override the template."""
        self._state = 0.0

    def advance(self, dt: float, forcingAmpl: float) -> float:
        """Override the template."""
        self._state = self._state + dt
        return dt

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state):
        """Override the template."""
        self._state = state

    def score(self):
        """Override the template."""
        return min(self._state * 10.0, 1.0)

    def noise(self):
        """Override the template."""
        return 0.0

    @classmethod
    def name(cls):
        """Return the model name."""
        return "SimpleFModel"


class DoubleWellModel(ForwardModel):
    """2D double well forward model.

    V(x,y) = x^4/4 - x^2/2 + y^2

    Associated SDE:
    dX_t = -nabla V(X_t)dt + g(X_t)dW_t

    with:
    -nabla V(X_t) = [x - x^3, -2y]

    With the 2 wells at [-1.0, 0.0] and [1.0, 0.0]
    """

    def __init__(self,
                 params: Optional[dict] = None,
                 ioprefix: Optional[str] = None):
        """Override the template."""
        self._state = self.initCondition()
        self._slow_factor = params.get("model",{}).get("slow_factor",0.00001)

    def __RHS(self, state):
        """Double well RHS function."""
        sleepTime = float(self._slow_factor * np.random.rand(1).item())
        time.sleep(sleepTime)
        return np.array([state[0] - state[0] ** 3, -2 * state[1]])

    def __dW(self, dt):
        """Stochastic forcing."""
        self._rand = np.random.randn(2)
        return np.sqrt(dt) * self._rand

    def initCondition(self):
        """Return the initial conditions."""
        return np.array([-1.0, 0.0])

    def advance(self, dt: float, forcingAmpl: float) -> float:
        """Override the template."""
        self._state = (
            self._state + dt * self.__RHS(self._state) + forcingAmpl * self.__dW(dt)
        )
        return dt

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state):
        """Override the template."""
        self._state = state

    def score(self):
        """Override the template."""
        a = np.array([-1.0, 0.0])
        b = np.array([1.0, 0.0])
        vA = self._state - a
        vB = self._state - b
        da = np.sum(vA**2, axis=0)
        db = np.sum(vB**2, axis=0)
        f1 = 0.5
        f2 = 1.0 - f1
        return f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)

    def noise(self):
        """Override the template."""
        return self._rand

    @classmethod
    def name(cls):
        """Return the model name."""
        return "DoubleWellModel"
