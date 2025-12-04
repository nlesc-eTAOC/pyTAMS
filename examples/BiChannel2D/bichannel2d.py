import contextlib
import typing
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pytams.database import Database
from pytams.fmodel import ForwardModelBaseClass


class BiChannel2D(ForwardModelBaseClass):
    """An implementation of the bi-channel.

    This is an overdamped dynamics, with two local minima connected
    to another by two channels: the upper one going through a local
    minimum at (0,1.5), and the lower one going through the saddle
    point around (0,-0.5).

    The minima are round A=(x_A,y_A)=(-1,0) and B (x_B,y_B)=(1,0).

    The system potential is given by:

    V(x,y) = 1/5 x^4 + 1/5 (y - 1/3)^4 + 3e^{-x^2-(y-1/3)^2}
            - 3e^{-x^2-(y-5/3)^2} - 5e^{-(x-1)^2-y^2} - 5e^{-(x+1)^2-y^2}
    """

    def _init_model(self, m_id: int, params: dict[typing.Any, typing.Any]) -> None:
        """Concrete class specific initialization.

        Args:
            m_id: the model instance unique identifier
            params: an optional dict containing parameters
        """
        # State a numpy array and initial state
        # close to A
        self._state = np.array([-0.9, 0.1])
        self._inv_T = params.get("model", {}).get("inv_temperature", 6.67)
        if params["model"]["deterministic"]:
            self._rng = np.random.default_rng(m_id)
        else:
            self._rng = np.random.default_rng()

    @classmethod
    def potential(cls, x: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Potential function.

        Args:
            x: the model state

        Returns:
            The bi-channel potential
        """
        return (
            1.0 / 5.0 * x[0] ** 4
            + 1.0 / 5.0 * (x[1] - 1.0 / 3.0) ** 4
            + 3.0 * np.exp(-(x[0] ** 2) - (x[1] - 1.0 / 3.0) ** 2)
            - 3.0 * np.exp(-(x[0] ** 2) - (x[1] - 5.0 / 3.0) ** 2)
            - 5.0 * np.exp(-((x[0] - 1.0) ** 2) - x[1] ** 2)
            - 5.0 * np.exp(-((x[0] + 1.0) ** 2) - x[1] ** 2)
        )

    @classmethod
    def drift(cls, x: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Drift function.

        The drift function f = - nabla(V)

        Args:
            x: the model state

        Returns:
            The bi-channel potential divergence
        """
        return np.array(
            [
                -4.0 / 5.0 * x[0] ** 3
                + 6.0 * x[0] * np.exp(-(x[0] ** 2) - (x[1] - 1.0 / 3.0) ** 2)
                - 6.0 * x[0] * np.exp(-(x[0] ** 2) - (x[1] - 5.0 / 3.0) ** 2)
                - 10.0 * (x[0] - 1.0) * np.exp(-((x[0] - 1.0) ** 2) - x[1] ** 2)
                - 10.0 * (x[0] + 1.0) * np.exp(-((x[0] + 1.0) ** 2) - x[1] ** 2),
                -4.0 / 5.0 * (x[1] - 1.0 / 3.0) ** 3
                + 6.0 * (x[1] - 1.0 / 3.0) * np.exp(-(x[0] ** 2) - (x[1] - 1.0 / 3.0) ** 2)
                - 6.0 * (x[1] - 5.0 / 3.0) * np.exp(-(x[0] ** 2) - (x[1] - 5.0 / 3.0) ** 2)
                - 10.0 * x[1] * np.exp(-((x[0] - 1.0) ** 2) - x[1] ** 2)
                - 10.0 * x[1] * np.exp(-((x[0] + 1.0) ** 2) - x[1] ** 2),
            ]
        )

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
        self._state = self._state + dt * self.drift(self._state) + np.sqrt(2 * dt / self._inv_T) * noise
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
        da = np.sqrt(np.sum((self._state - a) ** 2, axis=0))
        dab = np.sqrt(np.sum((a - b) ** 2, axis=0))
        return da / dab

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
        return "BiChannel2D"


def plot_in_landscape(fmodel: BiChannel2D, tdb: Database, idx: int) -> None:
    """Wrapper function to plot TAMS ensemble."""
    # Get a potential map
    x_range = np.linspace(-1.6, 1.6, 101)
    y_range = np.linspace(-1.6, 2.6, 132)
    x, y = np.meshgrid(x_range, y_range)
    potential = np.zeros((101, 132))
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            potential[i, j] = fmodel.potential(np.array([x_range[i], y_range[j]]))

    with contextlib.suppress(Exception):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    plt.figure(figsize=(6, 4))
    ctr = plt.contourf(x, y, np.transpose(potential), levels=25, cmap="viridis_r")

    for t in tdb.traj_list():
        state_x = np.array([s[1][0] for s in t.get_state_list()])
        state_y = np.array([s[1][1] for s in t.get_state_list()])
        plt.plot(state_x, state_y, color="k", alpha=0.6)

    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    plt.xlabel("x", fontsize="x-large")
    plt.ylabel("y", fontsize="x-large")
    plt.colorbar(ctr, label="V(x,y)")
    plt.grid(linestyle=":", color="silver")
    plt.tight_layout()
    plt.savefig(f"./Path_run_{idx:04d}.png", dpi=1200)
    plt.close()
