import typing
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from pytams.fmodel import ForwardModelBaseClass


class Doublewell2DDisk(ForwardModelBaseClass):
    """2D double well forward model.

    This version of the double well model state is a file
    on disk instead of the numpy array. This is closer
    to the behavior of high dimensional models.
    Note that the model still stores the state in memory
    from one step to the next.

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
        if not params.get("database", {}).get("path"):
            err_msg = "Database path needed for disk-based 2D double well model."
            raise RuntimeError(err_msg)

        self._db_path = self._workdir.parents[1]
        if not Path(self._workdir).exists():
            Path(self._workdir).mkdir()

        # The state is a str, the state_data a numpy array
        self._state_data = None
        self._state = self.init_condition()
        self._epsilon = params.get("model", {}).get("epsilon", 1.0)
        if params["model"]["deterministic"]:
            self._rng = np.random.default_rng(m_id)
        else:
            self._rng = np.random.default_rng()

    def init_condition(self) -> str:
        """Return the initial conditions.

        In this model, a path to the initial state is returned.
        """
        state_file = "init_state.npy"
        state_path = Path(self._workdir / state_file)
        self._state_data = np.array([-1.0, 0.0])
        np.save(state_path, self._state_data)
        return state_path.relative_to(self._db_path).as_posix()

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

    def _advance(self, step: int, _time: float, dt: float, noise: Any, need_end_state: bool) -> float:
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
        self._state_data = self._state_data + dt * self.drift(self._state_data) + np.sqrt(dt * self._epsilon) * noise
        if need_end_state:
            state_file = f"state_{step + 1:06}.npy"
            state_path = Path(self._workdir / state_file)
            self._state = state_path.relative_to(self._db_path).as_posix()
            np.save(state_path, self._state_data)
        else:
            self._state = None
        return dt

    def get_current_state(self) -> str:
        """Return the current state of the model.

        Note that the return type is left to the concrete model definition.
        """
        return self._state

    def set_current_state(self, state: str) -> None:
        """Set the current state of the model.

        Args:
            state: the externally provide state
        """
        self._state = state
        state_path = Path(self._db_path / self._state)
        self._state_data = np.load(state_path)

    def score(self) -> float:
        """Return the model's current state score.

        The score is a real.

        Returns:
            the score associated with the current model state
        """
        a = np.array([-1.0, 0.0])
        b = np.array([1.0, 0.0])
        da = np.sum((self._state_data - a) ** 2, axis=0)
        db = np.sum((self._state_data - b) ** 2, axis=0)
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
        return "Doublewell2DDisk"
