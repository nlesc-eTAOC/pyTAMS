"""A short script to perform hosing runs."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import toml
from boussinesq2d import Boussinesq2D
from pyrevs.core import Config
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory import TrajectoryConfig
from pyrevs.utils import setup_logger

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    fmodel = Boussinesq2D
    with Path("input_hosing.toml").open("r") as f:
        input_params = toml.load(f)

    setup_logger(loglevel="INFO")

    # Setup parameters
    cfg = Config(input_params)
    tcfg = cfg.load(TrajectoryConfig)
    model_params = cfg.section_dict("model")

    traj = Trajectory(traj_id=0, weight=1.0, fmodel_t=fmodel, traj_cfg=tcfg, model_params=model_params)
    traj.advance(t_end=120.0)
    traj.store(Path("./hysteresis_traj.xml"))

    plt.plot(traj.get_time_array(), traj.get_score_array())
    plt.grid()
    plt.show()
