import time
from typing import Any
from typing import Optional
import numpy as np
from pytams.fmodel import ForwardModelBaseClass
from pytams.tams import TAMS


class OrnsteinUlhenbeck(ForwardModelBaseClass):
    """A time-series Ornstein-Ulhenbeck process model.

    SDE:
        dX_t = -theta X_t dt + sqrt(2 * epsilon)  dW_t
    """

    def _init_model(self,
                    params: dict,
                    ioprefix: Optional[str] = None):
        """Init the OU model."""
        self._state = 0.0
        self._theta = params.get("model",{}).get("theta",1.0)
        self._epsilon = params.get("model",{}).get("epsilon",0.5)
        self._sigma = np.sqrt(self._epsilon/self._theta)
        self._target = params.get("model",{}).get("target",1*self._sigma)
        self._tau = 1.0 / self._theta
        self._rng = np.random.default_rng()

    def _advance(self,
                 step: int,
                 time: float,
                 dt: float,
                 noise: Any,
                 need_end_state: bool) -> float:
        """Advance the OU model."""
        # Advance using Euler-Maruyama explicit scheme
        noise_scaling = np.sqrt(dt)
        self._state = (1.0 - self._theta * dt) * self._state + self._sigma * noise * noise_scaling
        return dt

    def get_current_state(self):
        """Override the template."""
        return self._state

    def set_current_state(self, state):
        """Override the template."""
        self._state = state

    def score(self):
        """Override the template."""
        score = 1.0 - (self._target - self._state) / self._target
        return score

    def make_noise(self):
        """Noise generator."""
        # A single normal(0,1) random number
        return self._rng.standard_normal(1)

    @classmethod
    def name(cls):
        """Return the model name."""
        return "OrnsteinUlhenbeckModel"

if __name__ == "__main__":
    fmodel = OrnsteinUlhenbeck
    nRuns = 100
    proba = []
    return_time = []
    for _ in range(nRuns):
        tams = TAMS(fmodel_t=fmodel)
        proba.append(tams.compute_probability())
        print(f"Transition probability: {proba[-1]}")
    avg_proba = sum(proba) / nRuns
    var_proba = 0.0
    for i in range(len(proba)):
        var_proba = var_proba + (proba[i] - avg_proba)**2
    stdev_proba = np.sqrt(var_proba / (nRuns))
    var_proba = var_proba / (nRuns - 1)
    print(f"Averaged transition probability: {avg_proba}, stdev: {stdev_proba}, err: {np.sqrt(var_proba)/avg_proba}")
