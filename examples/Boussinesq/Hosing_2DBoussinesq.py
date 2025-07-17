import logging
from pathlib import Path
import numpy as np
import toml
import matplotlib.pyplot as plt
from TAMS_2DBoussinesq import Boussinesq2DModel
from pytams.trajectory import Trajectory

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fmodel = Boussinesq2DModel
    with open("input_hosing.toml", "r") as f:
        input_params = toml.load(f)

    traj = Trajectory(0, fmodel, input_params)
    traj.advance()
    traj.store(Path("./hysteresis_traj.xml"))

    plt.plot(traj.get_time_array(), traj.get_score_array())
    plt.grid()
    plt.show()
