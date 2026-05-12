import typing
from typing import Any
import numpy as np
import numpy.typing as npt
from pytams.core import ForwardModelBaseClass
from pytams.core import Snapshot


class Doublewell2D(ForwardModelBaseClass):
    """2D double well forward model.

    V(x,y) = x^4/4 - x^2/2 + y^2

    Associated SDE:
    dX_t = -nabla V(X_t)dt + g(X_t)dW_t

    with:
    -nabla V(X_t) = [x - x^3, -2y]
    g(X_t) = sqrt(epsilon)

    With the 2 wells at [-1.0, 0.0] and [1.0, 0.0]
    """

    def _init_model(self, m_id: int, params: dict[typing.Any, typing.Any]) -> None:
        """Concrete class specific initialization.

        Args:
            m_id: the model instance unique identifier
            params: an optional dict containing parameters
        """
        self._state = np.array([-0.92, 0.0])
        self._epsilon = params.get("epsilon", 1.0)
        if self._deterministic:
            self._rng = np.random.default_rng(m_id)
        else:
            self._rng = np.random.default_rng()

    @classmethod
    def potential(cls, x: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Potential function.

        The potential function of the 2D double well

        Args:
            x: the model state

        Returns:
            The 2D double well potential
        """
        return 1.0 / 4.0 * x[0] ** 4 - 1.0 / 2.0 * x[0] ** 2 + x[1] ** 2

    @classmethod
    def drift(cls, x: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Drift function.

        The drift function f = - nabla(V)

        Args:
            x: the model state

        Returns:
            The 2D double well potential divergence
        """
        return np.array([x[0] - x[0] ** 3, -2 * x[1]])

    def _advance(self, _step: int, _time: float, dt: float, noise: Any, _need_end_state: bool) -> float:
        """Concrete class advance function.

        This is the model-specific advance function.

        Args:
            step: the current step counter
            time: the starting time of the advance call
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
            need_end_state: whether the step end state is needed
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        self._state = self._state + dt * self.drift(self._state) + np.sqrt(dt * self._epsilon) * noise
        return dt

    def get_current_state(self) -> npt.NDArray[np.number]:
        """Return the current state of the model.

        Note that the return type is left to the concrete model definition.
        """
        return self._state

    def set_current_state(self, state: npt.NDArray[np.number]) -> None:
        """Set the current state of the model.

        Args:
            state: the externally provide state
        """
        self._state = state

    def score(self) -> float:
        """Return the model's current state score.

        The score is a real.

        Returns:
            the score associated with the current model state
        """
        a = np.array([-1.0, 0.0])
        b = np.array([1.0, 0.0])
        da = np.sum((self._state - a) ** 2, axis=0)
        db = np.sum((self._state - b) ** 2, axis=0)
        f1 = 0.5
        f2 = 1.0 - f1
        return f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)

    def make_noise(self) -> npt.NDArray[np.number]:
        """Return the model's latest noise increment.

        Note that the noise type is left to the concrete model definition.

        Returns:
            The model next noise increment
        """
        return self._rng.standard_normal(2)

    # def check_termination(self, step: int, time: float, nstep_end: int, time_end: float, current_score: float) -> bool:
    #    """Check if the trajectory is terminated.

    #    This default implementation checks if the current time or
    #    step is below the provided end time and end step.
    #    This is proper when running TAMS sampling, but not AMS or other methods.

    #    Args:
    #        step: the current step counter
    #        time: the time of the simulation
    #        nstep_end: the maximum number of steps to advance
    #        time_end: the end time of the advance
    #        current_score: the current score
    #    """
    #    r_state_a = np.sqrt(np.sum(self._state**2))
    #    return self._state[0] <= -0.95

    def diagnostic_hook(
        self,
        dlabel: str,
        tid: int,
        score_level: float,
        old_snap: Snapshot,
        new_snap: Snapshot,
    ) -> Any:
        """Diagnostic hook.

        Args:
            dlabel: the label of the diagnostic calling the hook
            tid: the ID of the trjaectory calling
            score_level: the score level crossed and triggering the call
            old_snap: the snapshot at the beginning of the step
            new_snap: the snapshot at the end of the step
        """
        _, _ = dlabel, tid
        return old_snap.state[1] + (new_snap.score - score_level) * (new_snap.state[1] - old_snap.state[1])

    @classmethod
    def name(cls) -> str:
        """Return a the model name."""
        return "Doublewell2D"
