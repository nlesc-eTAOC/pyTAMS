import time
from typing import Any
import numpy as np
from pytams.fmodel import ForwardModelBaseClass
from pytams.tams import TAMS


class DoubleWellModel3D(ForwardModelBaseClass):
    """3D double well forward model.

    V(x,y,z) = x^4/4 - x^2/2 + y^2 + z^4

    Associated SDE:
    dX_t = -nabla V(X_t)dt + g(X_t)dW_t

    with:
    -nabla V(X_t) = [x - x^3, -2y, -4z^3]

    With the 2 wells at [-1.0, 0.0, 0.0] and [1.0, 0.0, 0.0]
    """

    def _init_model(self,
                    params: dict = None,
                    ioprefix: str = None):
        """Override the template."""
        self._state = self.initCondition()
        self._rng = np.random.default_rng()

    def __RHS(self, state):
        """Double well RHS function."""
        sleepTime = float(0.00001 * np.random.rand(1).item())
        time.sleep(sleepTime)
        return np.array([state[0] - state[0] ** 3,
                         -2 * state[1],
                         -4 * state[2] ** 3])

    def __dW(self, dt, rand):
        """Stochastic forcing."""
        return np.sqrt(dt) * rand

    def initCondition(self):
        """Return the initial conditions."""
        return np.array([-1.0, 0.0, 0.0])

    def _advance(self,
                step: int,
                dt: float,
                noise: Any,
                forcingAmpl: float) -> float:
        """Override the template."""
        self._state = (
                self._state + dt * self.__RHS(self._state) + forcingAmpl * self.__dW(dt, noise[:3])
        )
        self._prescribed_noise = False
        return dt

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state):
        """Override the template."""
        self._state = state

    def score(self):
        """Override the template."""
        a = np.array([-1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        vA = self._state - a
        vB = self._state - b
        da = np.sum(vA**2, axis=0)
        db = np.sum(vB**2, axis=0)
        f1 = 0.5
        f2 = 1.0 - f1
        return f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)

    def _make_noise(self):
        """Override the template."""
        return self._rng.standard_normal(10)

    @classmethod
    def name(self):
        """Return the model name."""
        return "DoubleWellModel"

if __name__ == "__main__":
    fmodel = DoubleWellModel3D
    nRuns = 1
    proba = []
    for _ in range(nRuns):
        tams = TAMS(fmodel_t=fmodel)
        proba.append(tams.compute_probability())
        print("Transition probability: {}".format(proba[-1]))
    avg_proba = sum(proba) / nRuns
    var_proba = 0.0
    for prob in proba:
        var_proba = var_proba + (prob - avg_proba)**2
    var_proba = var_proba / (nRuns)
    print("Avg trans: {}, var: {}".format(avg_proba,var_proba))
