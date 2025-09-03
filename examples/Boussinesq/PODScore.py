"""A container class for POD-based score function."""
import logging
import numpy as np
import netCDF4 as nc
from pathlib import Path

_logger = logging.getLogger(__name__)

def project_in_POD_space(field, phi_POD, weights):
    """ Computes POD coefficients of a snapshot
    Parameters
    ----------
        field : np array
            Snapshot to project [nfield,depth,lat]
        phi_POD : np array
            POD basis functions [nmodes,nfield,depth,lat]
        weights : np array
            Spatial weights for projection [depth,lat]
    Returns
    -------
        field_POD : np array
            POd coefficients of snapshot field [nmodes]
    """

    nmodes = phi_POD.shape[0]

    # Take weighed inner product to get projection coefficients onto POD basis
    field_POD = np.zeros(nmodes)
    for i in range(nmodes):
        field_POD[i] = np.sum(field[:,:,:]*phi_POD[i,:,:,:]*weights[None,:,:])

    # Return POD coefficients
    return field_POD

def compute_score_function(field_POD, ref_psi_POD):
    """ Computes score function
    Parameters
    ----------
        field_POD: np array
            POD coefficients of field snapshot [nmodes]
        ref_psi_POD: np array
            Reference trajectory POD coefficients [ntime, nmodes]
    Returns
    -------
        score: float
            score value
    """

    # Get length of trajectory data
    ntime = ref_psi_POD.shape[0]

    # Computes the Euclidian distance in POD space
    # between snapshot and trajectory points
    dist = np.zeros(ntime)
    for t in range(ntime):
        dist[t] = np.sum((field_POD[:]-ref_psi_POD[t,:])**2)

    # Get closest index
    it = np.argmin(dist)
    # Computes penalty coefficient
    alpha = 1.0 # function of dist[it] in the future

    score = (it+1)/float(ntime)*alpha
    return score

class PODScore():
    """A class to hold POD data required for evaluating the score function."""
    def __init__(self,
                 lat_in: int,
                 depth_in: int,
                 pod_data_file: str) -> None:
        """Load the POD data and perform some checks.

        Args:
            lat_in: dimension in the latitude space
            depth_in: dimension in the depth space
            pod_data_file: the data containing the results of the POD decomposition
        """
        # User defined parameters
        self._nmodes = 8
        self._ntraj_ref_start = 3200
        self._ntraj_ref_end = 3800
        self._ntraj_start = 0
        self._ntraj_end = 119

        # Read POD decomposition data
        if not Path(pod_data_file).exists():
            err_msg = f"Could not find the {pod_data_file} POD decomposition data file !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        # Open data file and metadata
        nc_POD_in = nc.Dataset(pod_data_file, mode="r")
        shapes = nc_POD_in["stream_phi"].shape
        self._ntimes = shapes[0]
        self._lat = shapes[1]
        self._depth = shapes[2]

        # Checks
        if not (self._lat == lat_in and self._depth == depth_in):
            err_msg = f"Model size does not match the POD database"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        # Load data
        self._psi_POD = nc_POD_in["psi"][:,self._ntimes-self._nmodes:self._ntimes]
        self._sigma_POD = nc_POD_in["sigma"][self._ntimes-self._nmodes:self._ntimes]
        self._psi_POD[:,:] *= np.sqrt(self._sigma_POD[None,:])
        self._phi_POD = np.zeros((self._nmodes,2,self._lat,self._depth))
        self._phi_POD[:,0,:,:] = nc_POD_in["stream_phi"][self._ntimes-self._nmodes:self._ntimes,:,:]
        self._phi_POD[:,1,:,:] = nc_POD_in["salt_phi"][self._ntimes-self._nmodes:self._ntimes,:,:]
        self._phi_POD[:,:,:,:] /= np.sqrt(self._sigma_POD[:,None,None,None])
        self._weights = nc_POD_in["spatial_weights"][:,:]
        self._scaling_stream = nc_POD_in["scaling_stream"][:]
        self._scaling_salt = nc_POD_in["scaling_salt"][:]

    def get_score(self, model_state: Any) -> float:
        """Compute the score function of the given model state.

        Args:
            The model state as a numpy array (fix typing)

        Returns:
            The score function associated with the input state
        """
        field = np.zeros((2,self._lat,self._depth))
        field[0,:,:] = model_state[3,:,:] / self._scaling_stream
        field[1,:,:] = model_state[1,:,:] / self._scaling_salt

        field_POD = project_in_POD_space(field, self._phi_POD, self._weights)

        return compute_score_function(field_POD, self._psi_POD[self._ntraj_ref_start:self._ntraj_ref_end,:])
