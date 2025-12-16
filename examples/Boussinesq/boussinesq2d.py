"""Boussinesq 2D pyTAMS concrete implementation."""
import logging
from pathlib import Path
from typing import Any
import netCDF4
import numpy as np
import scipy as sp
from boussinesq_core import BoussinesqCore
from podscore import PODScore
from pytams.fmodel import ForwardModelBaseClass

_logger = logging.getLogger(__name__)

def loadnc(statefile: Path, state_field: str):
    """Load data from NetCDF file."""
    dset = netCDF4.Dataset(statefile, mode="r")
    dset.set_auto_mask(False)
    try:
        state_arrays = dset[state_field][:, :, :]
    except:
        err_msg = f"Unable to locate {state_field} in netCFD file {statefile}"
        print(err_msg)
    dset.close()

    return state_arrays

def getncfields(statefile: Path) -> list[str]:
    """Load data from NetCDF file."""
    dset = netCDF4.Dataset(statefile, mode="r")
    flist = [k for k in dset.variables.keys()]
    dset.close()

    return flist


class Boussinesq2D(ForwardModelBaseClass):
    """A forward model for the 2D Boussinesq model.

    The computational grid is [horizontal, vertical] of size (M+1)x(N+1).
    Note that all the physical parameters of the Boussinesq model are not
    exposed here, but are hard-coded in the Boussinesq class.

    The core model state is a 3D numpy array of vorticity, salinity,
    temperature and streamfunction (4x(M+1)x(N+1)).

    The model state (explosed to pyTAMS) is a tuple with a path to
    a netCDF file and the name of the field in the file, but
    this class also keeps the last version of the state in memory.

    Additional attributes:
        _M: number of horizontal grid points
        _N: number of vertical grid points
        _eps: noise level
        _K: number of forcing modes
        _B: Boussinesq model
    """

    def _init_model(self, m_id: int, params: dict[Any, Any]) -> None:
        """Initialize the model."""
        # Parse parameters
        subparms = params.get("model", {})
        self._M = subparms.get("size_M", 40)  # Horizontals
        self._N = subparms.get("size_N", 80)  # Verticals
        self._eps = subparms.get("epsilon", 0.01)  # Noise level
        self._K = subparms.get("K", 7)  # Number of forcing modes = 2*K
        self._delta_stoch = subparms.get("delta_stoch", 0.05)  # Noise depth
        self._stop_noise_time = subparms.get("stop_noise", -1.0)

        # Hosing parameters
        self._hosing_shape = subparms.get("hosing_shape", "tanh")
        self._hosing_rate = subparms.get("hosing_rate", 0.0)
        self._hosing_start = subparms.get("hosing_start", 0.0)
        self._hosing_end = subparms.get("hosing_end", -1.0)
        self._hosing_start_val = subparms.get("hosing_start_val", 0.0)

        # Asymmetry parameter
        self._beta = 0.1

        # Load the ON and OFF conditions
        self._on = np.load("stateON_beta_0p1.npy", allow_pickle=True)
        self._off = np.load("stateOFF_beta_0p1.npy", allow_pickle=True)

        # Score function parameters
        self._score_builder = None
        self._score_method = subparms.get("score_method", "default")
        self._time_dep_score = subparms.get("score_time_dep", False)
        self._moving_on_state = subparms.get("moving_on_state", False)
        if self._score_method == "PODdecomp":
            self._pod_data_file = subparms.get("pod_data_file", None)
            self._score_pod_d0 = subparms.get("pod_d0", None)
            self._score_pod_ndim = subparms.get("pod_ndim", 8)

        self._edge_state_file = subparms.get("edge_state_file", None)

        if self._time_dep_score:
            self._score_tfinal = params.get("trajectory", {}).get("end_time", 0.001)
            self._score_tscale = subparms.get("score_time_scale", 1.0)

        self._initialize_score_function()

        # Initialize random number generator
        # If deterministic run, set seed from the traj id
        if subparms["deterministic"]:
            self._rng = np.random.default_rng(m_id)
        else:
            self._rng = np.random.default_rng()

        # Initialize the Boussinesq model
        dt = params.get("trajectory", {}).get("step_size", 0.001)
        self._B = BoussinesqCore(self._M, self._N, dt)
        self._B.make_salinity_forcing(self._beta)
        self._B.init_salt_stoch_noise(self._B.zz, self._K, self._eps, self._delta_stoch)
        self._B.init_hosing(
            self._hosing_shape, self._hosing_start, self._hosing_end, self._hosing_start_val, self._hosing_rate
        )

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
        return "Boussinesq2D"

    def init_condition(self) -> tuple[str, str]:
        """Return the initial conditions."""
        # Set the initial state to the ON state
        self._state_arrays = self._on

        if not self._netcdf_state_path.exists():
            err_msg = f"Attempting to add data to {self._netcdf_state_path} file: it is missing !"
            _logger.error(err_msg)
            raise RuntimeError

        dset = netCDF4.Dataset(self._netcdf_state_path, mode="r+")
        state_data = dset.createVariable(f"state_{0:06}", np.float32, ("var", "lat", "depth"))
        state_data[:, :, :] = self._state_arrays
        dset.close()

        return (self._netcdf_state_path.relative_to(self._db_path).as_posix(), f"state_{0:06}")

    def init_storage(self) -> None:
        """Initialize a netCDF file to store state data in."""
        # Set Path and checks
        state_file = "states.nc"
        self._netcdf_state_path = Path(self._workdir / state_file)

        if self._netcdf_state_path.exists():
            return

        dset = netCDF4.Dataset(self._netcdf_state_path, mode="w", format="NETCDF4_CLASSIC")
        _ = dset.createDimension("var", 4)
        _ = dset.createDimension("lat", self._M + 1)
        _ = dset.createDimension("depth", self._N + 1)
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
            self._state_arrays = dset[state[1]][:, :, :]
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
                state_data = dset.createVariable(f"state_{step + 1:06}", np.float32, ("var", "lat", "depth"))
            except:
                err_msg = f"Attempting to overwrite state_{step + 1:06} in netCFD file {self._netcdf_state_path}"
                _logger.exception(err_msg)
                raise

            state_data[:, :, :] = self._state_arrays
            dset.close()
            self._state = (self._netcdf_state_path.relative_to(self._db_path).as_posix(), f"state_{step + 1:06}")

        return dt

    def _initialize_score_function(self) -> None:
        """Initialize the data for the score function."""
        # Dealing with non-autonomous forcing:
        # we consider that the ON state moves over time, let's
        # reload it periodically
        self._step_on_state = -1
        self._on_state_file = "stateON.nc"
        if self._moving_on_state:
            if Path(self._on_state_file).exists():
                self._on_state_dict = {}
                for f in getncfields(Path(self._on_state_file)):
                    fidx = int(f[-6:])
                    self._on_state_dict[fidx] = loadnc(Path(self._on_state_file), f)
                self._on_old = self._on_state_dict[0]
                self._on_new = self._on_state_dict[49]
                self._on = self._on_old
            else:
                wrn_msg = f"Moving A state activated but the state file {self._on_state_file} not found!"
                _logger.warning(wrn_msg)

        if self._score_method == "default":
            self._psi_north_on = np.mean(self._on[3, 28:34, 34:46], axis=(0, 1))
            self._psi_north_off = np.mean(self._off[3, 28:34, 34:46], axis=(0, 1))
        elif self._score_method == "BaarsJCP":
            self._edge_state = np.load(self._edge_state_file, allow_pickle=True)
            self._on_to_off_l2norm = np.sqrt(np.sum((self._on[1:3, :, :] - self._off[1:3, :, :]) ** 2))
            self._score_eta = (
                np.sqrt(np.sum((self._edge_state[1:3, :, :] - self._on[1:3, :, :]) ** 2)) / self._on_to_off_l2norm
            )


    def _update_on_state_data(self, step:int) -> None:
        """Load a new ON state from file."""
        # Return if we are not using a moving ON state
        if not self._moving_on_state:
            return

        # If OLD and NEW state are still good
        # do the interpolation
        if step < (self._step_on_state + 50):
            fnew = (step - self._step_on_state) / 50.0
            fold = 1.0 - fnew
            self._on = fnew * self._on_new + fold * self._on_old
        # Otherwise, update old and new
        else:
            self._step_on_state = step
            self._on_old = self._on_new
            self._on_new = self._on_state_dict[step]
            self._on = self._on_old

        # Update score-formulation specific data
        if self._score_method == "default":
            self._psi_north_on = np.mean(self._on[3, 28:34, 34:46], axis=(0, 1))
            self._psi_north_off = np.mean(self._off[3, 28:34, 34:46], axis=(0, 1))
        elif self._score_method == "BaarsJCP":
            self._on_to_off_l2norm = np.sqrt(np.sum((self._on[1:3, :, :] - self._off[1:3, :, :]) ** 2))
            self._score_eta = (
                np.sqrt(np.sum((self._edge_state[1:3, :, :] - self._on[1:3, :, :]) ** 2)) / self._on_to_off_l2norm
            )

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

        # Update the ON state
        self._update_on_state_data(self._step)

        xi_zero = None

        if self._score_method == "default":
            psi_north = np.mean(self._state_arrays[3, 28:34, 34:46], axis=(0, 1))

            xi_zero = (np.sqrt((psi_north - self._psi_north_on)**2.0) /
                       np.sqrt((self._psi_north_off - self._psi_north_on)**2.0))

        if self._score_method == "PODdecomp":
            if self._score_builder is None:
                self._score_builder = PODScore(
                    self._M + 1, self._N + 1, self._pod_data_file, self._score_pod_ndim, self._score_pod_d0
                )

            if self._moving_on_state:
                xi_zero = self._score_builder.get_score(self._state_arrays, self._on)
            else:
                xi_zero = self._score_builder.get_score(self._state_arrays)

        if self._score_method == "BaarsJCP":
            da = np.sqrt(np.sum((self._state_arrays[1:3, :, :] - self._on[1:3, :, :]) ** 2)) / self._on_to_off_l2norm
            db = np.sqrt(np.sum((self._state_arrays[1:3, :, :] - self._off[1:3, :, :]) ** 2)) / self._on_to_off_l2norm

            xi_zero = (
                self._score_eta
                - self._score_eta * np.exp(-8.0 * da**2)
                + (1.0 - self._score_eta) * np.exp(-8.0 * db**2)
            )

        # A score for the edge tracking algorithm
        if self._score_method == "EdgeTracker":
            # Compute the score and return -1, 0 or 1:
            # -1 : if the score get  below 0.1 (close to the ON state)
            #  1 : if the score gets above 0.9 (close to the OFF state)
            #  0 : otherwise
            psi_south = np.mean(self._state_arrays[3, 5:15, 32:48], axis=(0, 1))
            score = (psi_south - self._psi_south_on) / (self._psi_south_off - self._psi_south_on)
            if score > 0.9:
                return 1.0

            if score < 0.1:
                return -1.0

            return 0.0

        if xi_zero is None:
            err_msg = f"Unknown score method {self._score_method} !"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        # Compute an exponential decay near the time horizon of the
        # simulation final time if requested
        if self._time_dep_score:
            if self._time < self._stop_noise_time:
                return  xi_zero * (1.0 - np.exp((self._time - self._stop_noise_time)
                                   / self._score_tscale) * max(0.0, 0.5 - xi_zero)/0.5)

        return xi_zero

    def _trajectory_branching_hook(self) -> None:
        """Model-specific post trajectory branching hook."""
        # Set the moving on state
        if not self._moving_on_state:
            return

        lkeys = list(self._on_state_dict.keys())
        for k in range(len(lkeys)-1):
            if (self._step >= lkeys[k] and
                    self._step < lkeys[k+1]):
                self._step_on_state = lkeys[k]
                if self._step_on_state == 0:
                    self._step_on_state = -1
                self._on_old = self._on_state_dict[lkeys[k]]
                self._on_new = self._on_state_dict[lkeys[k+1]]
                fnew = (self._step - self._step_on_state) / 50.0
                fold = 1.0 - fnew
                self._on = fnew * self._on_new + fold * self._on_old
                break

    def check_convergence(self, step: int, time: float, current_score: float, target_score: float) -> bool:
        """Check if the model has converged.

        This is almost the default implementation, but if we are
        running the edge tracking algorithm exit when either
        ON or OFF state is reached.

        Args:
            step: the current step counter
            time: the time of the simulation
            current_score: the current score
            target_score: the target score
        """
        _ = (step, time)
        if self._score_method == "EdgeTracker":
            # We converged if the score if no longer zero
            return abs(current_score) > 0.5

        return current_score >= target_score

    def make_noise(self) -> Any:
        """Return a random noise.

        The model parameter stop_noise allows
        to return zero noise past a given time.
        """
        if self._stop_noise_time > 0.0 and self._time > self._stop_noise_time:
            return np.zeros(2 * self._K)

        return self._rng.normal(0, 1, size=(2 * self._K))
