"""Top-level sampler object."""

import logging
from typing import Any
from pytams.config import Config
from pytams.config import SystemConfig
from pytams.config import RuntimeConfig
from pytams.database import Database
from pytams.strategies.base_strategy import BaseSamplingStrategy
from pytams.utils import setup_logger

_logger = logging.getLogger(__name__)


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
        _strategy (BaseSamplingStrategy): The sampling strategy

    The configuration file is also passed to the logging and strategy setup routine.
    """

    def __init__(self, fmodel_t: Any,
                 sys_cfg: SystemConfig,
                 strategy: BaseSamplingStrategy,
                 database: Database) -> None:
        """Initialize a Sampler object.

        This constructor loads configuration parameters, initializes logging,
        instanciate the sampling strategy and prepares the sampling database.

        Args:
            fmodel_t: the forward model type
            sys_cfg: the system configuration
            strategy: the sampling strategy
            database: the sampling database

        Raises:
            ValueError: if the input file is not found
        """
        # Keep the fmodel_t around for now
        self._fmodel_t = fmodel_t

        # Load sampler parameters and setup logger
        self._runtime_cfg: RuntimeConfig = sys_cfg.runtime
        setup_logger(self._runtime_cfg.loglevel, self._runtime_cfg.logfile)

        # Instanciate sampling strategy
        self._strategy = strategy

        # Prepare diagnostics parameters
        # Note that diagnostic parameters are left as dictionaries
        # of Config (and not dataclasses) at this point
        self._diag_dicts: dict[str, Config] | None = None
        self._prepare_diagnostics()

        # Setup database
        self._db = database

    def _setup_db(self) -> None:
        """Initialize the sampling database.

        This method delegates database creation to the sampling strategy
        via ``BaseSamplingStrategy.initialize_db``.

        Notes:
            The structure and contents of the database are strategy-dependent.
            The resulting object is stored internally as ``self._db`` and passed
            unchanged to the strategy during sampling.
        """
        self._db = self._strategy.initialize_db(self._diag_dicts)

    def _prepare_diagnostics(self) -> None:
        """Extract the diagnostics parameters from the root config."""
        if len(self._runtime_cfg.diagnostics) > 0:
            self._diag_dicts = {}
            for diag in self._runtime_cfg.diagnostics:
                self._diag_dicts[diag] = self._config.section(diag)

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
        inf_msg = f"Starting rare event sampling with {self._strategy} with walltime = {self._runtime_cfg.walltime} s"
        _logger.info(inf_msg)

        self._strategy.sample(self._db, self._runtime_cfg.walltime, self._runtime_cfg.plot_diagnostics)

    @property
    def database(self) -> Database:
        """Access the sampling database."""
        return self._db
