"""A short script to perform hosing runs."""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import toml
from boussinesq2d import Boussinesq2D
from pytams.trajectory import Trajectory

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fmodel = Boussinesq2D
    with Path("input_hosing.toml").open("r") as f:
        input_params = toml.load(f)

    traj = Trajectory(0, 1.0, fmodel, input_params)
    traj.advance()
    traj.store(Path("./hysteresis_traj.xml"))

    plt.plot(traj.get_time_array(), traj.get_score_array())
    plt.grid()
    plt.show()
