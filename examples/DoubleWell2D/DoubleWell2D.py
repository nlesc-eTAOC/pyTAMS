import typing
from typing import Any
import numpy as np
import numpy.typing as npt
from pytams.fmodel import ForwardModelBaseClass


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
        self._state = np.array([-1.0, 0.0])
        self._epsilon = params.get("model", {}).get("epsilon", 1.0)
        if params["model"]["deterministic"]:
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

    @classmethod
    def name(cls) -> str:
        """Return a the model name."""
        return "Doublewell2D"
