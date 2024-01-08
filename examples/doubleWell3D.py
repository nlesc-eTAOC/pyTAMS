import time
import numpy as np
from pytams.fmodel import ForwardModel
from pytams.tams import TAMS


class DoubleWellModel3D(ForwardModel):
    """3D double well forward model.

    V(x,y,z) = x^4/4 - x^2/2 + y^2 + z^4

    Associated SDE:
    dX_t = -nabla V(X_t)dt + g(X_t)dW_t

    with:
    -nabla V(X_t) = [x - x^3, -2y, -4z^3]

    With the 2 wells at [-1.0, 0.0, 0.0] and [1.0, 0.0, 0.0]
    """

    def __init__(self, params: dict = None, ioprefix: str = None):
        """Override the template."""
        self._state = self.initCondition()

    def __RHS(self, state):
        """Double well RHS function."""
        sleepTime = float(0.001 * np.random.rand(1).item())
        time.sleep(sleepTime)
        return np.array([state[0] - state[0] ** 3,
                         -2 * state[1],
                         -4 * state[2] ** 3])

    def __dW(self, dt):
        """Stochastic forcing."""
        return np.sqrt(dt) * np.random.randn(3)

    def initCondition(self):
        """Return the initial conditions."""
        return np.array([-1.0, 0.0, 0.0])

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
        a = np.array([-1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        vA = self._state - a
        vB = self._state - b
        da = np.sum(vA**2, axis=0)
        db = np.sum(vB**2, axis=0)
        f1 = 0.5
        f2 = 1.0 - f1
        return f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)

    @classmethod
    def name(self):
        """Return the model name."""
        return "DoubleWellModel"

if __name__ == "__main__":
    fmodel = DoubleWellModel3D
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 400,
        "Verbose": True,
        #"DB_save": True,
        #"DB_prefix": "DW_3DTest",
        "DB_restart": "DW_3DTest.tdb_Final",
        "dask.nworker": 1,
        "wallTime": 200.0,
        "traj.end_time": 10.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.95,
        "traj.stoichForcing": 0.4,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    print("Transition probability: {}".format(transition_proba))
