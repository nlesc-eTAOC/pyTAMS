import os
import time
from pathlib import Path
from typing import Any
import numpy as np
from pytams.fmodel import ForwardModelBaseClass
from pytams.tams import TAMS


class DoubleWellModel3DDisk(ForwardModelBaseClass):
    """3D double well forward model.

    This version of the double well model state is a file
    on disk instead of the numpy array. This is closer
    to the behavior of high dimensional models.
    Note that the model still store the state in memory
    from one step to the next.


    V(x,y,z) = x^4/4 - x^2/2 + y^2 + z^4

    Associated SDE:
    dX_t = -nabla V(X_t)dt + g(X_t)dW_t

    with:
    -nabla V(X_t) = [x - x^3, -2y, -4z^3]

    With the 2 wells at [-1.0, 0.0, 0.0] and [1.0, 0.0, 0.0]
    """

    def _init_model(self,
                    params: dict,
                    ioprefix: str = None):
        """Initialize the Double Well Model Disk.

        Args:
            params: a dict containing parameters
            ioprefix: an optional string defining run folder
        """
        if not params.get("database",{}).get("path"):
            raise RuntimeError("Database path needed for disk-based model.")

        self._db_path = self._workdir.parents[1]
        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir)

        self._state_data = None
        self._state = self.initCondition()
        self._slow_factor = params.get("model",{}).get("slow_factor",0.00000001)
        self._noise_amplitude = params.get("model",{}).get("noise_amplitude",1.0)
        if params["model"]["deterministic"]:
            seed = int(ioprefix[4:10])
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()


    def __RHS(self, state):
        """Double well RHS function."""
        sleepTime = float(self._slow_factor * np.random.rand(1).item())
        time.sleep(sleepTime)
        return np.array([state[0] - state[0] ** 3,
                         -2 * state[1],
                         -4 * state[2] ** 3])

    def __dW(self, dt, rand):
        """Stochastic forcing."""
        return np.sqrt(dt) * rand

    def initCondition(self):
        """Return the initial conditions.

        In this model, a path to the initial state is returned.
        """
        state_file = "init_state.npy"
        state_path = Path(self._workdir / state_file)
        self._state_data = np.array([-1.0, 0.0, 0.0])
        np.save(state_path, self._state_data)
        return state_path.relative_to(self._db_path).as_posix()

    def _advance(self,
                 step: int,
                 time: float,
                 dt: float,
                 noise: Any,
                 need_end_state: bool) -> float:
        """Override the template."""
        new_state = (
                self._state_data + dt * self.__RHS(self._state_data) + self._noise_amplitude * self.__dW(dt, noise[:3])
        )
        self._state_data = new_state
        if need_end_state:
            state_file = f"state_{step+1:06}.npy"
            state_path = Path(self._workdir / state_file)
            self._state = state_path.relative_to(self._db_path).as_posix()
            np.save(state_path, self._state_data)
        else:
            self._state = None
        return dt

    def get_current_state(self):
        """Override the template."""
        return self._state

    def set_current_state(self, state):
        """Override the template."""
        self._state = state
        state_path = Path(self._db_path / self._state)
        self._state_data = np.load(state_path)

    def score(self):
        """Override the template."""
        a = np.array([-1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        vA = self._state_data - a
        vB = self._state_data - b
        da = np.sum(vA**2, axis=0)
        db = np.sum(vB**2, axis=0)
        f1 = 0.5
        f2 = 1.0 - f1
        return f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)

    def make_noise(self):
        """Override the template."""
        return self._rng.standard_normal(3)

    @classmethod
    def name(self):
        """Return the model name."""
        return "DoubleWellModel3DDisk"

if __name__ == "__main__":
    fmodel = DoubleWellModel3DDisk
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
