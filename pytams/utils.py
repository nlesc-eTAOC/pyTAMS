"""A set of utility functions for TAMS."""

import ast
import inspect
import logging
import sys
import textwrap
from abc import ABCMeta
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt

_logger = logging.getLogger(__name__)

def is_windows_os() -> bool:
    """Indicates Windows platform."""
    system = sys.platform.lower()
    return system.startswith("win")

def is_mac_os() -> bool:
    """Indicates MacOS platform."""
    system = sys.platform.lower()
    return system.startswith("dar")

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
        maxes: array of maximas across all trajectories
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
    # or from the 'examples' folder of pyTAMS
    return [
        m
        for m in all_modules
        if (
            hasattr(sys.modules[m], "__file__")
            and (
                Path(str(sys.modules[m].__file__)).parent == Path().absolute()
                or any((p.name == "examples") for p in Path(str(sys.modules[m].__file__)).parents)
            )
        )
    ]


def generate_subclass(abc_cls: ABCMeta, class_name: str, file_path: str) -> None:
    """Generate a subclass skeleton.

    Implementing all abstract methods from `abc_cls`, written to `file_path`.

    Args:
        abc_cls: an ABC
        class_name: the new subclass name
        file_path: where to write the subclass
    """
    # Identify abstract methods
    abstract_methods = {
        name: value for name, value in abc_cls.__dict__.items() if getattr(value, "__isabstractmethod__", False)
    }

    # Build import line
    module_name = abc_cls.__module__
    abc_name = abc_cls.__name__
    import_lines = [f"from {module_name} import {abc_name}\n", "import typing\n", "from typing import Any\n\n\n"]

    # Build class header
    lines = [*import_lines, f"class {class_name}({abc_name}):\n"]
    lines.append('    """TODO: add class docstring."""\n')

    if not abstract_methods:
        lines.append("    pass\n")
    else:
        # Generate each required method with preserved signature
        for name, func in abstract_methods.items():
            sig = inspect.signature(func)
            doc = inspect.getdoc(func)

            lines.append(f"    def {name}{sig}:\n")
            if doc:
                # Indent docstring correctly
                doc_clean = textwrap.indent('"""' + doc + '\n"""', " " * 8)
                lines.append(f"{doc_clean}\n")
            else:
                lines.append('        """TODO: implement method."""\n')

            lines.append("        # Implement concrete method body\n\n")

    # Add the name class method
    lines.append("    @classmethod\n")
    lines.append("    def name(cls) -> str:\n")
    lines.append('        """Return a the model name."""\n')
    lines.append(f'        return "{class_name}"\n')

    # Write to file
    Path(file_path).write_text("".join(lines))
