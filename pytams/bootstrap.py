import argparse
from pathlib import Path
import toml
from pytams.config import Config
from pytams.config import SystemConfig
from pytams.database import Database
from pytams.fmodel import ForwardModelBaseClass
from pytams.sampler import RareEventSampler
from pytams.strategies.base_strategy import BaseSamplingStrategy


def parse_cl_args(a_args: list[str] | None = None) -> argparse.Namespace:
    """Parse provided list or default CL argv.

    Args:
        a_args: optional list of options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input .toml file", default="input.toml")
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


def build_strategy(
    sys_cfg: SystemConfig,
    fmodel_t: type[ForwardModelBaseClass],
) -> BaseSamplingStrategy:
    """Instantiate the sampling strategy.

    Args:
        sys_cfg: Fully resolved system configuration
        fmodel_t: Forward model type

    Returns:
        Concrete BaseSamplingStrategy instance
    """
    strategy_name = sys_cfg.sampler.strategy

    return BaseSamplingStrategy.create(
        strategy_name,
        fmodel_t=fmodel_t,
        runtime_cfg=sys_cfg.runtime,
        deterministic=sys_cfg.sampler.deterministic,
        strategy_cfg=sys_cfg.strategy,
    )

def build_database(
    fmodel_t: type[ForwardModelBaseClass],
    cfg: Config,
    sys_cfg: SystemConfig,
    strategy: BaseSamplingStrategy,
) -> Database:
    """Instantiate and initialize the database.

    Responsibilities:
    - Create DB object
    - Let strategy define DB structure
    - Persist full resolved configuration

    Args:
        fmodel_t: Forward model type
        cfg: Raw Config (still useful for flexible sections like [model])
        sys_cfg: Fully resolved SystemConfig
        strategy: Sampling strategy instance

    Returns:
        Initialized Database
    """
    db = Database(
        fmodel_t=fmodel_t,
        config=cfg,
        strategy=strategy,
    )

    # Let strategy define DB schema/content
    strategy.setup_database(db)

    # Persist FULL resolved config (including defaults)
    if db.path() is not None:
        sys_cfg.write_toml(Path(db.path()) / "input_params.toml")

    return db

def build_sampler(
    fmodel_t: type[ForwardModelBaseClass],
    sys_cfg: SystemConfig,
    strategy: BaseSamplingStrategy,
    database: Database,
) -> RareEventSampler:
    """Instantiate the top-level sampler.

    Args:
        fmodel_t: Forward model type
        sys_cfg: Fully resolved SystemConfig
        strategy: Sampling strategy
        database: Initialized database

    Returns:
        Ready-to-run RareEventSampler
    """
    return RareEventSampler(
        fmodel_t=fmodel_t,
        sys_cfg=sys_cfg,
        strategy=strategy,
        database=database,
    )
