import logging
from pathlib import Path
from typing import Any
import numpy as np
import scipy as sp
from Boussinesq_2DAMOC import Boussinesq
from pytams.fmodel import ForwardModelBaseClass
from pytams.tams import TAMS

_logger = logging.getLogger(__name__)


class Boussinesq2DModel(ForwardModelBaseClass):
    """A forward model for the 2D Boussinesq model.

    The computational grid is [horizontal, vertical] of size (M+1)x(N+1).
    Note that all the physical parameters of the Boussinesq model are not
    exposed here, but are hard-coded in the Boussinesq class.

    The model state is a 3D numpy array of vorticity, salinity,
    temperature and streamfunction (4x(M+1)x(N+1)).

    The model state (returned to TAMS) is a path to a numpy file, but
    this class also keeps the last version of the state in memory.

    Additional attributes:
        _M: number of horizontal grid points
        _N: number of vertical grid points
        _eps: noise level
        _K: number of forcing modes
        _B: Boussinesq model
    """

    def _init_model(self, params: dict | None = None, ioprefix: str | None = None) -> None:
        """Initialize the model."""
        # Parse parameters
        subparms = params.get("model", {})
        self._M = subparms.get("size_M", 40)  # Horizontals
        self._N = subparms.get("size_N", 80)  # Verticals
        self._eps = subparms.get("epsilon", 0.05)  # Noise level
        self._hosing_rate = subparms.get("hosing_rate", 0.0)
        self._hosing_start = subparms.get("hosing_start", 0.0)
        self._hosing_start_val = subparms.get("hosing_start_val", 0.0)
        self._K = subparms.get("K", 4)  # Number of forcing modes = 2*K

        # Initialize random number generator
        # If deterministic run, set seed from the traj id
        if subparms["deterministic"]:
            self._rng = np.random.default_rng(int(ioprefix[4:10]))
        else:
            self._rng = np.random.default_rng()

        # Load the ON and OFF conditions
        # The 140th is with Beta = 0.1
        n_select = 140
        self._beta_span = np.load("beta.npy", allow_pickle=True)[n_select]
        self._on = np.load("stateON_beta_0p1.npy", allow_pickle=True)
        self._off = np.load("stateOFF_beta_0p1.npy", allow_pickle=True)
        self._psi_south_on = np.mean(self._on[3, 5:15, 32:48], axis=(0, 1))
        self._psi_south_off = np.mean(self._off[3, 5:15, 32:48], axis=(0, 1))

        # Initialize the Boussinesq model
        dt = params.get("trajectory", {}).get("step_size", 0.001)
        self._B = Boussinesq(self._M, self._N, dt)
        self._B.make_FS(self._beta_span)
        self._B.init_Snoise(self._B.zz, self._K, self._eps)
        self._B.init_hosing(self._hosing_start, self._hosing_start_val, self._hosing_rate)

        # Initial conditions from ON state
        # Create the workdir if it doesn't exist
        self._db_path = self._workdir.parents[1]
        if not self._workdir.exists():
            self._workdir.mkdir(parents=True)

        # History
        self._history = []

        # Keep the last state data in-memory
        self._state_arrays = None

        # The state is a path to a npy file on disk
        self._state = self.init_condition()

    @classmethod
    def name(cls) -> str:
        """Return the model name."""
        return "2DBoussinesqModel"

    def init_condition(self) -> str:
        """Return the initial conditions."""
        state_file = "init_state.npy"
        state_path = Path(self._workdir / state_file)
        np.save(state_path, self._on)
        self._state_arrays = self._on
        return state_path.relative_to(self._db_path).as_posix()

    def get_current_state(self) -> Any:
        """Access the model state."""
        return self._state

    def set_current_state(self, state: Any) -> None:
        """Set the model state.

        And load the state from disk into _state_arrays

        Args:
            state: the new state
        """
        self._state = state
        state_path = self._db_path.joinpath(self._state)
        self._state_arrays = np.load(state_path)

    def _advance(self, step: int, time: float, dt: float, noise: Any, need_end_state: bool) -> float:
        """Advance the model.

        See J. Soons et al. for details on the model PDEs.

        Args:
            step: the current step counter
            time: the starting time of the advance call
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
            need_end_state: whether the step end state is needed

        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        # Construct the full 2D stoch. noise from
        # mode amplitude
        full_noise = self._B.Snoise(noise)

        # Load individual component from the state
        if self._state_arrays is None:
            err_msg = f"Model state is not initialized while calling advance at time {time}"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)
        w_old, sal_old, temp_old, psi_old = self._state_arrays

        fx_psi = self._B.Fx @ psi_old
        psi_fz = psi_old @ self._B.FzT

        # Temperature, Salinity updates
        adv_s, adv_t = fx_psi[np.newaxis] * (self._state_arrays[1:3] @ self._B.DzT) - psi_fz[np.newaxis] * (
            self._B.Dx @ self._state_arrays[1:3]
        )
        rhs_temp = temp_old + dt * (adv_t + self._B.FT)
        rhs_sal = sal_old + dt * (adv_s + self._B.FS + self._B.get_hosing(time)) + np.sqrt(dt) * full_noise

        temp_new = sp.linalg.solve_sylvester(self._B.AT, self._B.BT, rhs_temp)
        sal_new = sp.linalg.solve_sylvester(self._B.AS, self._B.BS, rhs_sal)

        # Normalize the salinity
        sal_0 = 1.0
        sal_new *= sal_0 / np.mean(sal_new)

        # Vorticity update
        src_w = self._B.Pr * self._B.Ra * self._B.Dx @ (temp_new - sal_new) @ self._B.S_corr
        adv_w = fx_psi * (w_old @ self._B.FzT) - psi_fz * (self._B.Fx @ w_old)
        rhs_w = w_old + dt * (adv_w + src_w)
        w_new = sp.linalg.solve_sylvester(self._B.Aw, self._B.Bw, rhs_w)

        # Streamfunction update with the poisson operator
        psi_new = sp.linalg.solve_sylvester(self._B.Fxx, self._B.FzzT, -w_new)

        # Update the state
        self._state_arrays = np.array([w_new, sal_new, temp_new, psi_new])
        if need_end_state:
            state_file = f"state_{step + 1:06}.npy"
            state_path = Path(self._workdir / state_file)
            np.save(state_path, self._state_arrays)
            self._state = state_path.relative_to(self._db_path).as_posix()

        return dt

    def score(self) -> float:
        """Compute the score function.

        The current score function is a nomalized distance between the ON
        and OFF states in the stream function space (specifically the
        mean streamfunction in the southern ocean).

        Return:
            the score
        """
        if self._state_arrays is None:
            err_msg = "Model state is not initialized while calling advance"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)
        psi_south = np.mean(self._state_arrays[3, 5:15, 32:48], axis=(0, 1))

        return (psi_south - self._psi_south_on) / (self._psi_south_off - self._psi_south_on)

    def make_noise(self) -> Any:
        """Return a random noise."""
        return self._rng.normal(0, 1, size=(2 * self._K))


if __name__ == "__main__":
    fmodel = Boussinesq2DModel
    tams = TAMS(fmodel_t=fmodel)
    transition_proba = tams.compute_probability()
    print(f"Transition probability: {transition_proba}")
