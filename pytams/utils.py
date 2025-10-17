"""A set of utility functions for TAMS."""

import ast
import logging
import sys
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt

_logger = logging.getLogger(__name__)


def setup_logger(params: dict[Any, Any]) -> None:
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


def get_min_scored(maxes: npt.NDArray[Any], nworkers: int) -> tuple[list[int], npt.NDArray[Any]]:
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
    min_idx_list: list[int] = []
    for idx in ordered_tlist:
        if len(min_idx_list) > 0:
            is_same_min = maxes[idx] == maxes[min_idx_list[-1]]
        if len(min_idx_list) < nworkers or is_same_min:
            min_idx_list.append(int(idx))

    min_vals = maxes[min_idx_list]
    return min_idx_list, min_vals


def moving_avg(arr_in: npt.NDArray[Any], window_l: int) -> npt.NDArray[Any]:
    """Return the moving average of a 1D numpy array.

    Args:
        arr_in: 1D numpy array
        window_l: length of the moving average window

    Returns:
        1D numpy array
    """
    arr_out = np.zeros(arr_in.shape[0])
    for i in range(len(arr_in)):
        lbnd = max(i - int(np.ceil(window_l / 2)), 0)
        hbnd = min(i + int(np.floor(window_l / 2)), len(arr_in) - 1)
        if lbnd == 0:
            hbnd = window_l
        if hbnd == len(arr_in) - 1:
            lbnd = len(arr_in) - window_l - 1
        arr_out[i] = np.mean(arr_in[lbnd:hbnd])
    return arr_out


def get_module_local_import(module_name: str) -> list[str]:
    """Helper function getting local imported mods list.

    When pickling the forward model code, the model itself can import from
    several other local files. We also want to pickle those by value so let's get
    the list.

    Args:
        module_name: a module name we want the locally imported modules

    Returns:
        A list of local modules names imported within the provide module
    """
    # Check that module exists
    if module_name not in sys.modules:
        err_msg = f"Attempting to extract sub import from {module_name} missing from currently loaded modules"
        _logger.exception(err_msg)
        raise ValueError(err_msg)

    # Check access to the module file
    if hasattr(sys.modules[module_name], "__file__") and Path(str(sys.modules[module_name].__file__)).exists():
        mfile = Path(str(sys.modules[module_name].__file__))
    else:
        err_msg = f"Attempting to locate sub import file from {module_name}, but file is missing or undefined"
        _logger.exception(err_msg)
        raise FileNotFoundError(err_msg)

    # Parse the module file
    # for imports
    with mfile.open("r") as f:
        file_raw = f.read()

    file_ast = ast.parse(file_raw)
    all_modules = []

    for node in ast.walk(file_ast):
        # Append "import X" type
        if isinstance(node, ast.Import):
            all_modules.extend([x.name for x in node.names])
        # Append "from X import Y" type
        if isinstance(node, ast.ImportFrom) and node.module:
            all_modules.append(node.module)

    # Return only those whose file is in the current folder
    return [
        m
        for m in all_modules
        if (hasattr(sys.modules[m], "__file__") and Path(str(sys.modules[m].__file__)).parent == Path().absolute())
    ]
