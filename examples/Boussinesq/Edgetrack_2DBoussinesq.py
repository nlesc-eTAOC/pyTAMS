import logging
from pathlib import Path
import numpy as np
import toml
from TAMS_2DBoussinesq import Boussinesq2DModel
from pytams.trajectory import Trajectory
from edgetracking_algorithm import edgetracking

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fmodel = Boussinesq2DModel
    with open("input_edge.toml", "r") as f:
        input_params = toml.load(f)

    traj = Trajectory(0, fmodel, input_params)
    
    on_state = ()   # ON state array
    off_state = ()  # OFF state array

    upper, lower, edgetrack = edgetracking(on_state, off_state,
        eps1 = 1e-3,
        eps2 = 5e-3,
        maxiter = 10
    )

    # Save upper, lower, edgetrack as xr.DataArrays