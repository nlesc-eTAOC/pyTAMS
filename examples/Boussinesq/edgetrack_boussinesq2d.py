"""A short script to track edge state in Boussinesq model."""
import shutil
from pathlib import Path
import numpy as np
import toml
from boussinesq2d import Boussinesq2D
from edgetracking_algorithm import edgetracking
from pytams.utils import setup_logger

if __name__ == "__main__":
    fmodel = Boussinesq2D
    with Path("input_edge.toml").open("r") as f:
        input_params = toml.load(f)

    # Use pyTAMS internal logger setup
    setup_logger(input_params)

    on_state = np.load("stateON_beta_0p1.npy", allow_pickle=True)
    off_state = np.load("stateOFF_beta_0p1.npy", allow_pickle=True)

    # Create a temporary folder for the model
    Path("./.edge_tmp/").mkdir()

    try:
        upper, lower, edgetrack = edgetracking(
            fmodel, input_params, 0.1, on_state, off_state, eps1=1e-3, eps2=3e-3, maxiter=100, accuracy=1e-2
        )

    except:
        shutil.rmtree("./.edge_tmp")
        raise

    # Save edgetrack
    for k in range(len(edgetrack)):
        save_path = Path(f"edge_{k:04}.npy")
        np.save(save_path, edgetrack[k])
