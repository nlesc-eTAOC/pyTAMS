import logging
from pathlib import Path
import numpy as np
import toml
import shutil
from TAMS_2DBoussinesq import Boussinesq2DModel
from pytams.trajectory import Trajectory
from edgetracking_algorithm import edgetracking

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fmodel = Boussinesq2DModel
    with open("input_edge.toml", "r") as f:
        input_params = toml.load(f)

    on_state = np.load("stateON_beta_0p1.npy", allow_pickle=True)
    off_state = np.load("stateOFF_beta_0p1.npy", allow_pickle=True)

    # Create a temporary folder for the model
    Path("./.edge_tmp/").mkdir()

    upper, lower, edgetrack = edgetracking(fmodel, input_params, on_state, off_state,
        eps1 = 1e-3,
        eps2 = 5e-3,
        maxiter = 10
    )

    shutil.rmtree(f"./.edge_tmp")

    # Save upper, lower, edgetrack as xr.DataArrays
