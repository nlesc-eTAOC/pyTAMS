"""A configuration class to expose limited configuration."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
import toml
from pytams.config import Config
from pytams.config import RuntimeConfig
from pytams.config.core import collect_sections
from pytams.config.core import merge_config
from pytams.database import DatabaseConfig
from pytams.runner import RunnerConfig
from pytams.strategies.ams import AMSConfig
from pytams.strategies.montecarlo import MCConfig
from pytams.trajectory import TrajectoryConfig
from .config import SamplerConfig

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SystemConfig:
    """Overarching system configuration.

    This is a helper metadata class for the system configuration
    used to IO full configuration (i.e. including defaults)
    and performing configuration merging.
    """

    sampler: SamplerConfig
    runtime: RuntimeConfig
    strategy: AMSConfig | MCConfig
    database: DatabaseConfig
    runner: RunnerConfig
    trajectory: TrajectoryConfig

    @classmethod
    def __load__(cls, cfg: Config) -> SystemConfig:
        sampler = cfg.load(SamplerConfig)

        strategy_cls: type[AMSConfig | MCConfig]

        if sampler.strategy == "ams":
            strategy_cls = AMSConfig
        elif sampler.strategy == "montecarlo":
            strategy_cls = MCConfig
        else:
            err_msg = f"Unknown strategy '{sampler.strategy}'"
            raise ValueError(err_msg)

        return cls(
            sampler=sampler,
            runtime=cfg.load(RuntimeConfig),
            strategy=cfg.load(strategy_cls),
            database=cfg.load(DatabaseConfig),
            runner=cfg.load(RunnerConfig),
            trajectory=cfg.load(TrajectoryConfig),
        )

    def write_toml(self, path: Path, other_data: dict[str, Any]) -> None:
        """Write the system configuration to a TOML file."""
        data = collect_sections(
            self.sampler,
            self.runtime,
            self.strategy,
            self.database,
            self.runner,
            self.trajectory,
        )

        with path.open("w") as f:
            toml.dump(data | other_data, f)

    @classmethod
    def merge(
        cls,
        old: SystemConfig,
        new: SystemConfig,
    ) -> SystemConfig:
        """Merge two SystemConfig objects.

        Immutable sections must match exactly.
        Replaceable sections are overwritten by `new`.

        Args:
            old: Existing configuration (e.g. from database).
            new: Incoming configuration (e.g. from CLI/TOML).

        Returns:
            A merged SystemConfig instance.

        Raises:
            ValueError: If immutable sections differ.
        """
        return cls(
            sampler=merge_config(old.sampler, new.sampler),
            runtime=merge_config(old.runtime, new.runtime),
            strategy=merge_config(old.strategy, new.strategy),
            database=merge_config(old.database, new.database),
            runner=merge_config(old.runner, new.runner),
            trajectory=merge_config(old.trajectory, new.trajectory),
        )
