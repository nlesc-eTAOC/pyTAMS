"""Top-level sampler object."""

import argparse
import logging
from pathlib import Path
from typing import Any
import toml
from pytams.sampling_strategy import SamplingStrategy
from pytams.utils import setup_logger

_logger = logging.getLogger(__name__)


def parse_cl_args(a_args: list[str] | None = None) -> argparse.Namespace:
    """Parse provided list or default CL argv.

    Args:
        a_args: optional list of options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="pyTAMS input .toml file", default="input.toml")
    return parser.parse_args() if a_args is None else parser.parse_args(a_args)


class RareEventSampler:
    """The top-level rare event sampler object.

    A user-facing object to sample rare events of
    the forward model using a prescribed sampling strategy.
    """

    def __init__(self,
                 fmodel_t: Any,
                 strategy: SamplingStrategy,
                 a_args: list[str] | None = None) -> None:
        """Initialize a Sampler object.

        Args:
            strategy: the sampling strategy
            a_args: optional list of options

        Raises:
            ValueError: if the input file is not found
        """
        self._fmodel_t = fmodel_t

        input_file = vars(parse_cl_args(a_args=a_args))["input"]
        if not Path(input_file).exists():
            err_msg = f"Could not find the {input_file} pyREVS input file !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        with Path(input_file).open("r") as f:
            self._parameters = toml.load(f)

        # Setup logger
        setup_logger(self._parameters)

        # Time management uses UTC date
        # to make sure workers are always in sync
        self._wallTime: float = self._parameters.get("sampler", {}).get("walltime", 24.0 * 3600.0)

        # Store strategy
        self.strategy = strategy

        # Setup database
        self._setup_db()

    def _setup_db(self) -> None:
        """Create the database needed for the sampling strategy."""
        self._db = self.strategy.initialize_db()

    def run(self) -> None:
        """Sample rare events."""
        self.strategy.sample(self._db, self._wallTime)

    def set_strategy(self, new_strategy: SamplingStrategy) -> None:
        """Set a new sampling strategy."""
        self.strategy = new_strategy
