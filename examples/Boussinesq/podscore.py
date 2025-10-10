"""A container class for POD-based score function."""

import logging
from pathlib import Path
import netCDF4
import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_splprep

_logger = logging.getLogger(__name__)


def project_in_pod_space(
    nmodes: int, field: npt.NDArray[np.number], phi_pod: npt.NDArray[np.number], weights: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    """Computes POD coefficients of a snapshot.

    Args:
        nmodes: Number of modes to project on to
        field : Snapshot to project [nfield,depth,lat]
        phi_pod : POD basis functions [nmodes,nfield,depth,lat]
        weights : Spatial weights for projection [depth,lat]

    Returns:
        field_pod : np array
            POd coefficients of snapshot field [nmodes]
    """
    field_pod = np.zeros(nmodes)
    field_pod[:] = np.sum(field[None, :, :, :] * phi_pod[:nmodes, :, :, :] * weights[None, None, :, :], axis=(1, 2, 3))

    return field_pod


def compute_score_function(
    field_pod: npt.NDArray[np.number],
    ref_psi_pod: npt.NDArray[np.number],
    curv_abs: npt.NDArray[np.number],
    curvature: npt.NDArray[np.number],
    d0: float | None = None,
) -> float:
    """Computes score function using curvilinear abs.

    Args:
        field_pod: POD coefficients of field snapshot [nmodes]
        ref_psi_pod: Reference trajectory POD coefficients [ntime, nmodes]
        curv_abs: The curvilinear abcissa along the reference traj [ntime]
        curvature: The curvature of the reference trajectory
        d0: The exponential decay distance parameter

    Returns:
        score: float
            score value
    """
    # Get length of trajectory data
    ntime = ref_psi_pod.shape[0]

    # Computes the Euclidian distance in POD space
    # between snapshot and trajectory points
    dist = np.zeros(ntime)
    dist[:] = np.sqrt(np.sum((field_pod[None, :] - ref_psi_pod[:, :]) ** 2, axis=1))

    # Get closest index
    it = np.argmin(dist)

    if it > 0 and it < len(dist) - 1:
        # Get next adjacent index
        it_next = it + 1 if dist[it + 1] <= dist[it - 1] else it - 1

        # dot product
        dot = np.sum((field_pod[:] - ref_psi_pod[it, :]) * (ref_psi_pod[it_next, :] - ref_psi_pod[it, :]))
        segment_norm = np.sum(
            (ref_psi_pod[it_next, :] - ref_psi_pod[it, :]) * (ref_psi_pod[it_next, :] - ref_psi_pod[it, :])
        )

        frac = dot / segment_norm if np.abs(dot / segment_norm) <= 1.0 else 0.0
        closest_point = ref_psi_pod[it, :] + frac * (ref_psi_pod[it_next, :] - ref_psi_pod[it, :])
        distance = np.sum((field_pod[:] - closest_point[:]) ** 2)
        distance = np.sqrt(distance)
        curv = curv_abs[it] + frac * (curv_abs[it_next] - curv_abs[it])
        curvat = curvature[it] + frac * (curvature[it_next] - curvature[it])

    else:
        distance = dist[it]
        curv = curv_abs[it]
        curvat = 1.0

    # Computes penalty coefficient
    if d0:
        alpha = 1.0 * np.exp(-(distance**2) / d0**2)
    else:
        inv_curvature = max(0.02, 1.0 / curvat)
        alpha = 1.0 * np.exp(-(distance**2) / inv_curvature**2)

    return curv * alpha


class PODScore:
    """A class to hold POD data required for evaluating the score function."""

    def __init__(
        self,
        lat_in: int,
        depth_in: int,
        pod_data_file: str,
        score_space_dim: int = 8,
        score_d0: float | None = None,
        nsample: int = 200,
    ) -> None:
        """Load the POD data and perform some checks.

        Args:
            lat_in: dimension in the latitude space
            depth_in: dimension in the depth space
            pod_data_file: the data containing the results of the POD decomposition
            score_space_dim: the dimension of the low-dim space to get the score
            score_d0: the decay distance from the original data traj
            nsample: the time dimension after resampling the reference trajectory
        """
        # User-defined parameters
        self._n_active_modes = score_space_dim
        self._d0 = score_d0
        self._nsample = nsample

        # Read POD decomposition data
        if not Path(pod_data_file).exists():
            err_msg = f"Could not find the {pod_data_file} POD decomposition data file !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        # Open data file and metadata
        nc_pod_in = netCDF4.Dataset(pod_data_file, mode="r")
        self._ntimes = nc_pod_in.dimensions["time"].size
        self._lat = nc_pod_in.dimensions["lat"].size
        self._depth = nc_pod_in.dimensions["depth"].size
        self._nmodes = nc_pod_in.dimensions["mode"].size

        # Checks
        if not (self._lat == lat_in and self._depth == depth_in):
            err_msg = "Model size does not match the POD database"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        if self._n_active_modes > self._nmodes:
            err_msg = f"Requested {self._n_active_modes} but only {self._nmodes} available !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        # Load data
        self._psi_pod = nc_pod_in["psi"][:, :]
        self._sigma_pod = nc_pod_in["sigma"][:]
        self._psi_pod[:, :] *= np.sqrt(self._sigma_pod[None, :])
        self._phi_pod = np.zeros((self._nmodes, 2, self._lat, self._depth))
        self._phi_pod[:, 0, :, :] = nc_pod_in["stream_phi"][:, :, :]
        self._phi_pod[:, 1, :, :] = nc_pod_in["salt_phi"][:, :, :]
        self._phi_pod[:, :, :, :] /= np.sqrt(self._sigma_pod[:, None, None, None])
        self._weights = nc_pod_in["spatial_weights"][:, :]
        self._scaling_stream = nc_pod_in["scaling_stream"][:]
        self._scaling_salt = nc_pod_in["scaling_salt"][:]

        # Resample and smooth the reference trajectory
        self.resample_psi_pod_spline(n_uniform=nsample)

        # Compute the abscissa
        self.compute_curvilinear_abs()

        # Compute the trajectory curvature
        self.compute_curvature()

    def resample_psi_pod_spline(self, n_uniform: int = 200, smoothing: float = 0.01) -> None:
        """Resample the POD psi using Bsplines.

        Args:
            n_uniform: the number of points after resampling
            smoothing: a smoothing factor to remove kinks from the trajectory
        """
        # Fit spline
        tck, _ = make_splprep(self._psi_pod.T, s=smoothing)

        # Uniform parameterization
        t_new = np.linspace(0, 1, n_uniform)
        self._psi_pod = tck(t_new).T

    def compute_curvilinear_abs(self) -> None:
        """Compute the curvilinear abscissa for getting the score."""
        self._curv_abs = np.zeros(self._nsample)
        for k in range(1, self._nsample):
            self._curv_abs[k] = self._curv_abs[k - 1] + np.sqrt(
                np.sum((self._psi_pod[k, : self._n_active_modes] - self._psi_pod[k - 1, : self._n_active_modes]) ** 2)
            )

        self._curv_abs = self._curv_abs / self._curv_abs[-1]

    def compute_curvature(self) -> None:
        """Compute curvature."""
        # First derivative wrt s
        dr_ds = np.gradient(self._psi_pod, self._curv_abs, axis=0)

        # Second derivative wrt s
        d2r_ds2 = np.gradient(dr_ds, self._curv_abs, axis=0)

        # Curvature: norm of second derivative
        curvature = np.linalg.norm(d2r_ds2, axis=1)

        # Handle beginning and end of the trajectory
        lcurv = curvature.shape[0]
        portion = 10
        targ_c = 1.0

        loc_c = curvature[int(lcurv / portion)]
        curvature[: int(lcurv / portion)] = loc_c + (
            1.0 - self._curv_abs[: int(lcurv / portion)] / self._curv_abs[int(lcurv / portion)]
        ) * (targ_c - loc_c)

        loc_c = curvature[-int(lcurv / portion)]
        curvature[lcurv - int(lcurv / portion) : lcurv] = targ_c + (
            1.0 - self._curv_abs[lcurv - int(lcurv / portion) : lcurv]
        ) / (1.0 - self._curv_abs[-int(lcurv / portion)]) * (loc_c - targ_c)

        self._curvature = curvature

    def get_score(self, model_state: npt.NDArray[np.number]) -> float:
        """Compute the score function of the given model state.

        Args:
            model_state: The model state as a numpy array (fix typing)

        Returns:
            The score function associated with the input state
        """
        field = np.zeros((2, self._lat, self._depth))
        field[0, :, :] = model_state[3, :, :] / self._scaling_stream
        field[1, :, :] = model_state[1, :, :] / self._scaling_salt

        field_pod = project_in_pod_space(self._n_active_modes, field, self._phi_pod, self._weights)

        return compute_score_function(
            field_pod, self._psi_pod[:, : self._n_active_modes], self._curv_abs, self._curvature, d0=self._d0
        )
