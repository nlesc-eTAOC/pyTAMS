"""Top-level sampler object."""

import argparse
import logging
from pathlib import Path
from typing import Any
import toml
from pytams.base_strategy import BaseSamplingStrategy
from pytams.config import Config
from pytams.database import Database
from pytams.utils import setup_logger

_logger = logging.getLogger(__name__)


def parse_cl_args(a_args: list[str] | None = None) -> argparse.Namespace:
    """Parse provided list or default CL argv.

    Args:
        a_args: optional list of options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input .toml file", default="input.toml")
    return parser.parse_args() if a_args is None else parser.parse_args(a_args)


class RareEventSampler:
    """The top-level interface for rare event sampling.

    This class provides a user-facing entry point to perform rare event
    sampling using a specified :class:`BaseSamplingStrategy`.
    It is responsible for:

    - Parsing configuration from a TOML input file
    - Initializing logging
    - Instanciating the proper sampling strategy
    - Managing global runtime parameters (e.g., walltime)
    - Setting up the database
    - Running the sampling strategy

    Attributes:
        _config (Config): Configuration parameters parsed from the input file
        _wallTime (float): Maximum runtime in seconds
        _plot_diags (bool): Enable diagnostic plots during sampling
        _strategy (BaseSamplingStrategy): The sampling strategy

    Notes:
    The sampler assumes that the input file contains a ``[sampler]`` section
    with optional fields:

    - ``walltime`` (float): Maximum runtime in seconds (default: 24h)
    - ``plot_diagnostics`` (bool): Enable diagnostic plotting during sampling
    - ``strategy`` (str): The name of the sampling strategy

    The configuration file is also passed to the logging and strategy setup routine.
    """

    def __init__(self, fmodel_t: Any, a_args: list[str] | None = None) -> None:
        """Initialize a Sampler object.

        This constructor loads configuration parameters, initializes logging,
        instanciate the sampling strategy and prepares the sampling database.

        Args:
            fmodel_t: the forward model type
            a_args: optional list of options

        Raises:
            ValueError: if the input file is not found
        """
        input_file = vars(parse_cl_args(a_args=a_args))["input"]
        if not Path(input_file).exists():
            err_msg = f"Could not find the {input_file} pyREVS input file !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        with Path(input_file).open("r") as f:
            self._config = Config(toml.load(f))

        # Setup logger
        sampler_cfg = self._config.section("sampler", {})
        setup_logger(sampler_cfg.get("loglevel", "info"), sampler_cfg.get("logfile", None))

        # Time management uses UTC date
        # to make sure workers are always in sync
        # A 24h default is set
        self._wallTime: float = sampler_cfg.get("walltime", 24.0 * 3600.0)
        if not isinstance(self._wallTime, (int, float)):
            err_msg = "sampler.walltime must be a number"
            _logger.exception(err_msg)
            raise TypeError(err_msg)

        # Enable/disable diagnostic plots during sampling
        self._plot_diags = sampler_cfg.get("plot_diagnostics", False)

        # Instanciate sampling strategy
        strategy_type = sampler_cfg.get("strategy", "ams")
        if not isinstance(strategy_type, str):
            err_msg = "sampler.strategy must be a string"
            _logger.exception(err_msg)
            raise TypeError(err_msg)
        self._strategy: BaseSamplingStrategy = BaseSamplingStrategy.create(
            strategy_type, fmodel_t=fmodel_t, config=self._config
        )

        # Setup database
        self._setup_db()

    def _setup_db(self) -> None:
        """Initialize the sampling database.

        This method delegates database creation to the sampling strategy
        via ``BaseSamplingStrategy.initialize_db``.

        Notes:
            The structure and contents of the database are strategy-dependent.
            The resulting object is stored internally as ``self._db`` and passed
            unchanged to the strategy during sampling.
        """
        self._db = self._strategy.initialize_db()

    def run(self) -> None:
        """Execute the rare event sampling procedure.

        This method starts the sampling process by delegating execution to
        the configured ``BaseSamplingStrategy``.

        Notes:
            This method is typically the main entry point after initialization.
            At this point, it does not return a value; results are expected to be stored in the
            database or written to disk by the strategy.
            Future extensions will allow to perform several runs (possibly in parallel)
        """
        inf_msg = f"Starting rare event sampling with {self._strategy} with walltime = {self._wallTime} s"
        _logger.info(inf_msg)

        self._strategy.sample(self._db, self._wallTime, self._plot_diags)

    @property
    def database(self) -> Database:
        """Access the sampling database."""
        return self._db
