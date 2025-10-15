import logging
from pathlib import Path
from typing import Any
import numpy as np
import scipy as sp
import netCDF4
from Boussinesq_2DAMOC import Boussinesq
from podscore import PODScore
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
        self._eps = subparms.get("epsilon", 0.01)  # Noise level
        self._K = subparms.get("K", 7)  # Number of forcing modes = 2*K
        self._delta_stoch = subparms.get("delta_stoch", 0.05) # Noise depth

        # Hosing parameters
        self._hosing_rate = subparms.get("hosing_rate", 0.0)
        self._hosing_start = subparms.get("hosing_start", 0.0)
        self._hosing_start_val = subparms.get("hosing_start_val", 0.0)

        # Score function parameters
        self._score_builder = None
        self._score_method = subparms.get("score_method", "default")
        if self._score_method == "PODdecomp":
            self._pod_data_file = subparms.get("pod_data_file", None)
            self._score_pod_d0 = subparms.get("pod_d0", None)
            self._score_pod_ndim = subparms.get("pod_ndim", 8)

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
        self._B.make_salinity_forcing(self._beta_span)
        self._B.init_salt_stoch_noise(self._B.zz, self._K, self._eps, self._delta_stoch)
        self._B.init_hosing(self._hosing_start, self._hosing_start_val, self._hosing_rate)

        # Initial conditions from ON state
        # Create the workdir if it doesn't exist
        self._db_path = self._workdir.parents[1]
        if not self._workdir.exists():
            self._workdir.mkdir(parents=True)

        # Define (conceptually) the initial state
        # the file or data are not created yet
        self._state = (Path(self._workdir / "states.nc").relative_to(self._db_path).as_posix(), f"state_{0:06}")

        # Keep the last state data in-memory
        self._state_arrays = None

        # Keep around a flag to data init
        self._need_init_data = True

    @classmethod
    def name(cls) -> str:
        """Return the model name."""
        return "2DBoussinesqModel"

    def init_condition(self) -> (str,str):
        """Return the initial conditions."""
        # Set the initial state to the ON state
        self._state_arrays = self._on

        if not self._netcdf_state_path.exists():
            err_msg = f"Attempting to add data to {self._netcdf_state_path} file: it is missing !"
            _logger.error(err_msg)
            raise RuntimeError

        dset = netCDF4.Dataset(self._netcdf_state_path, mode="r+")
        state_data = dset.createVariable(f"state_{0:06}",np.float32,("var","lat","depth"))
        state_data[:,:,:] = self._state_arrays
        dset.close()

        return (self._netcdf_state_path.relative_to(self._db_path).as_posix(), f"state_{0:06}")

    def init_storage(self) -> str:
        """Initialize a netCDF file to store state data in."""
        # Set Path and checks
        state_file = "states.nc"
        self._netcdf_state_path = Path(self._workdir / state_file)

        if self._netcdf_state_path.exists():
            #wrn_msg = f"Attempting to create {self._netcdf_state_path} file: already present !"
            #_logger.warning(wrn_msg)
            return

        dset = netCDF4.Dataset(self._netcdf_state_path, mode="w",format="NETCDF4_CLASSIC")
        _ = dset.createDimension("var", 4)
        _ = dset.createDimension("lat", self._M+1)
        _ = dset.createDimension("depth", self._N+1)
        dset.close()


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
        state_path = self._db_path.joinpath(state[0])

        if not state_path.exists():
            err_msg = f"Attempting to read data from {state_path} file: it is missing !"
            _logger.error(err_msg)
            raise RuntimeError

        dset = netCDF4.Dataset(state_path, mode="r")
        dset.set_auto_mask(False)
        try:
            self._state_arrays = dset[state[1]][:,:,:]
        except:
            err_msg = f"Unable to locate {state[1]} in netCFD file {state_path}"
            _logger.exception(err_msg)
            raise
        dset.close()

        # If the model data were not initialized yet, create the container
        if self._need_init_data:
            # Initialize netCDF data container
            self.init_storage()

            self._need_init_data = False

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
        if self._need_init_data:
            # Initialize netCDF data container
            self.init_storage()

            # The state is a path to a netCDF file on disk
            # plus a variable name (a tuple)
            self._state = self.init_condition()

            self._need_init_data = False

        # Construct the full 2D stoch. noise from
        # mode amplitude
        full_noise = self._B.salt_stoch_noise(noise)

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
        src_w = self._B._pr * self._B._ra * self._B.Dx @ (temp_new - sal_new) @ self._B.S_corr
        adv_w = fx_psi * (w_old @ self._B.FzT) - psi_fz * (self._B.Fx @ w_old)
        rhs_w = w_old + dt * (adv_w + src_w)
        w_new = sp.linalg.solve_sylvester(self._B.Aw, self._B.Bw, rhs_w)

        # Streamfunction update with the poisson operator
        psi_new = sp.linalg.solve_sylvester(self._B.Fxx, self._B.FzzT, -w_new)

        # Update the state
        self._state_arrays = np.array([w_new, sal_new, temp_new, psi_new])
        if need_end_state:
            if not self._netcdf_state_path.exists():
                err_msg = f"Attempting to add data to {self._netcdf_state_path} file: it is missing !"
                _logger.error(err_msg)
                raise RuntimeError

            dset = netCDF4.Dataset(self._netcdf_state_path, mode="r+")

            try:
                state_data = dset.createVariable(f"state_{step + 1:06}",np.float32,("var","lat","depth"))
            except:
                err_msg = f"Attempting to overwrite state_{step + 1:06} in netCFD file {self._netcdf_state_path}"
                _logger.exception(err_msg)
                raise

            state_data[:,:,:] = self._state_arrays
            dset.close()
            self._state = (self._netcdf_state_path.relative_to(self._db_path).as_posix(), f"state_{step + 1:06}")

        return dt

    def score(self) -> float:
        """Compute the score function.

        The current score function accept one of two options:
         - a nomalized distance between the ON and OFF states in the
           stream function space (specifically the mean streamfunction
           in the southern ocean).
         - a score function based on the POD decomposition of the model
           dynamics.

        Return:
            the score
        """
        # If the model has not been initialized yet
        # simply return 0.0
        if self._need_init_data:
            return 0.0

        # Now if the model is None, something went wrong
        if self._state_arrays is None:
            err_msg = "Model state is empty while calling score"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        if self._score_method == "default":
            psi_south = np.mean(self._state_arrays[3, 5:15, 32:48], axis=(0, 1))

            return (psi_south - self._psi_south_on) / (self._psi_south_off - self._psi_south_on)

        if self._score_method == "PODdecomp":
            if self._score_builder is None:
                self._score_builder = PODScore(
                    self._M + 1, self._N + 1, self._pod_data_file, self._score_pod_ndim, self._score_pod_d0
                )

            return self._score_builder.get_score(self._state_arrays)

        err_msg = f"Unknown score method {self._score_method} !"
        _logger.exception(err_msg)
        raise RuntimeError(err_msg)

    def make_noise(self) -> Any:
        """Return a random noise."""
        return self._rng.normal(0, 1, size=(2 * self._K))


if __name__ == "__main__":
    fmodel = Boussinesq2DModel
    tams = TAMS(fmodel_t=fmodel)
    transition_proba = tams.compute_probability()
    print(f"Transition probability: {transition_proba}")
