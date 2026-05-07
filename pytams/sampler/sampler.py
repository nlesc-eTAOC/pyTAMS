"""Top-level sampler object."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any
import toml
from pytams.config import Config
from pytams.config import RuntimeConfig
from pytams.database import Database
from pytams.database import prepare_database_path
from pytams.strategies.base_strategy import BaseSamplingStrategy
from pytams.utils import setup_logger
from .system_config import SystemConfig

_logger = logging.getLogger(__name__)


def parse_cl_args(a_args: list[str] | None = None) -> argparse.Namespace:
    """Parse provided list or default CL argv.

    Args:
        a_args: optional list of options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input .toml file", default="input.toml")
    parser.add_argument(
        "-ov", "--overwrite", help="overwrite existing database input params", default=False, action="store_true"
    )
    return parser.parse_args() if a_args is None else parser.parse_args(a_args)


def load_config(path: Path) -> Config:
    """Load a TOML file into a Config object.

    Args:
        path: Path to TOML file

    Returns:
        Config instance wrapping the TOML data

    Raises:
        FileNotFoundError: if file does not exist
    """
    if not path.exists():
        err_msg = f"Config file not found: {path}"
        raise FileNotFoundError(err_msg)

    with path.open("r") as f:
        data = toml.load(f)

    return Config(data)


def build_system_config(cfg: Config) -> SystemConfig:
    """Build the fully resolved SystemConfig.

    This applies:
    - defaults
    - nested dataclass construction
    - strategy selection logic

    Args:
        cfg: Raw Config object

    Returns:
        Fully instantiated SystemConfig
    """
    return cfg.load(SystemConfig)


def build_database(fmodel_t: Any, cfg: Config, sys_cfg: SystemConfig, overwrite: bool) -> Database:
    """Build the database."""
    def_toml_output = "input_params.toml"
    # Instanciate the database
    # Load existing database if possible
    if sys_cfg.database.path and Path(sys_cfg.database.path).exists() and not sys_cfg.database.restart:
        # First build old system config
        db_cfg = load_config(Path(sys_cfg.database.path) / def_toml_output)
        db_sys_cfg = build_system_config(db_cfg)

        # Merge & update, or overwrite
        if overwrite:
            updated_sys_cfg = sys_cfg
            model_dict = cfg.section_dict("model")
            wrn_msg = "Overwriting existing database input parameters might cause issues!"
            _logger.warning(wrn_msg)
        else:
            updated_sys_cfg = SystemConfig.merge(db_sys_cfg, sys_cfg)
            model_dict = db_cfg.section_dict("model")
        updated_sys_cfg.write_toml(Path(sys_cfg.database.path) / def_toml_output, {"model": model_dict})

        # Load
        return Database.load(Path(sys_cfg.database.path), read_only=False)

    # Archive old database if present and restart is requested
    if sys_cfg.database.path:
        prepare_database_path(Path(sys_cfg.database.path), sys_cfg.database.restart)

    db = Database.create(fmodel_t, cfg)
    if sys_cfg.database.path:
        sys_cfg.write_toml(Path(sys_cfg.database.path) / def_toml_output, {"model": cfg.section_dict("model")})

    return db


def build_sampler(fmodel_t: Any, a_args: list[str] | None = None) -> RareEventSampler:
    """Instantiate the top-level sampler.

    Args:
        fmodel_t: Forward model type
        a_args: optional list of options

    Returns:
        Ready-to-run RareEventSampler
    """
    # Parse from command line to Config
    args = parse_cl_args(a_args)
    cfg = load_config(Path(args.input))

    # Build SystemConfig: typed input, apply defaults
    sys_cfg = build_system_config(cfg)

    # Instanciate the database
    db = build_database(fmodel_t, cfg, sys_cfg, args.overwrite)

    # Instanciate the strategy
    strategy = BaseSamplingStrategy.create(
        sys_cfg.sampler.strategy,
        fmodel_t=fmodel_t,
        runtime_cfg=sys_cfg.runtime,
        runner_cfg=sys_cfg.runner,
        strategy_cfg=sys_cfg.strategy,
        deterministic=sys_cfg.sampler.deterministic,
    )

    # Prepare diagnostics parameters
    # Note that diagnostic parameters are left as dictionaries
    # of Config (and not dataclasses) at this point
    diag_dicts: dict[str, Config] | None = None
    if len(sys_cfg.runtime.diagnostics) > 0:
        diag_dicts = {}
        for diag in sys_cfg.runtime.diagnostics:
            diag_dicts[diag] = cfg.section(diag)

    # Let strategy define DB schema/content
    strategy.initialize_database_schema(db, diag_dicts)

    return RareEventSampler(
        fmodel_t=fmodel_t,
        runtime_cfg=sys_cfg.runtime,
        strategy=strategy,
        database=db,
    )


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

    def __init__(
        self, fmodel_t: Any, runtime_cfg: RuntimeConfig, strategy: BaseSamplingStrategy, database: Database
    ) -> None:
        """Initialize a Sampler object.

        This constructor loads configuration parameters, initializes logging,
        instanciate the sampling strategy and prepares the sampling database.

        Args:
            fmodel_t: the forward model type
            runtime_cfg: the runtime configuration
            strategy: the sampling strategy
            database: the sampling database

        Raises:
            ValueError: if the input file is not found
        """
        # Keep the fmodel_t around for now
        self._fmodel_t = fmodel_t

        # Load sampler parameters and setup logger
        self._runtime_cfg = runtime_cfg
        setup_logger(self._runtime_cfg.loglevel, self._runtime_cfg.logfile)

        # Instanciate sampling strategy
        self._strategy = strategy

        # Setup database
        self._db = database

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
