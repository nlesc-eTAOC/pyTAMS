"""A set of utility functions for TAMS."""

import logging
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
