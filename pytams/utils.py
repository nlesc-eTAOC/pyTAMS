"""A set of utility functions for TAMS."""

import logging
import numpy.typing as npt
import numpy as np
from typing import Any


def setup_logger(params : dict[Any,Any]) -> None:
    """Setup the logger parameters.

    Args:
        params: a dictionary of parameters
    """
    # Set logging level
    log_level_str = params["tams"].get("loglevel", "INFO")
    if log_level_str.upper() == "DEBUG":
        log_level = logging.DEBUG
    elif log_level_str.upper() == "INFO":
        log_level = logging.INFO
    elif log_level_str.upper() == "WARNING":
        log_level = logging.WARNING
    elif log_level_str.upper() == "ERROR":
        log_level = logging.ERROR

    log_format = "[%(levelname)s] %(asctime)s - %(message)s"

    # Set root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
    )

    # Add file handler to root logger
    if params["tams"].get("logfile", None):
        log_file = logging.FileHandler(params["tams"]["logfile"])
        log_file.setLevel(log_level)
        log_file.setFormatter(logging.Formatter(log_format))
        logging.getLogger("").addHandler(log_file)

def get_min_scored(maxes : npt.NDArray[Any],
                   nworkers : int) -> tuple[list[int], npt.NDArray[Any]]:
        """Get the nworker lower scored trajectories or more if equal score.

        Args:
            maxes: array of maximas accros all trajectories
            nworkers: number of workers

        Returns:
            list of indices of the nworker lower scored trajectories
            array of minimas
        """
        ordered_tlist = np.argsort(maxes)
        is_same_min = False
        min_idx_list = []
        for idx in ordered_tlist:
          if len(min_idx_list) > 0:
            is_same_min = maxes[idx] == maxes[min_idx_list[-1]]
          if (len(min_idx_list) < nworkers or
              is_same_min):
            min_idx_list.append(idx)

        min_vals = maxes[min_idx_list]

        return min_idx_list, min_vals
