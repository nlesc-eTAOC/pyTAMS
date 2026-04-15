"""The main MonteCarlo class."""

import argparse
import logging
from pathlib import Path
from typing import Any
import toml
from pytams.database import Database
from pytams.sampling_strategy import SamplingStrategy
from pytams.taskrunner import get_runner_type
from pytams.utils import setup_logger
from pytams.worker import pool_worker

_logger = logging.getLogger(__name__)

STALL_TOL = 1e-10


def parse_cl_args(a_args: list[str] | None = None) -> argparse.Namespace:
    """Parse provided list or default CL argv.

    Args:
        a_args: optional list of options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="pyTAMS input .toml file", default="input.toml")
    return parser.parse_args() if a_args is None else parser.parse_args(a_args)


class MonteCarlo(SamplingStrategy):
    """A strategy class implementing MonteCarlo."""

    def __init__(self, fmodel_t: Any, a_args: list[str] | None = None) -> None:
        """Initialize a TAMS object.

        Args:
            fmodel_t: the forward model type
            a_args: optional list of options

        Raises:
            ValueError: if the input file is not found
        """
        self._fmodel_t = fmodel_t

        input_file = vars(parse_cl_args(a_args=a_args))["input"]
        if not Path(input_file).exists():
            err_msg = f"Could not find the {input_file} TAMS input file !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        with Path(input_file).open("r") as f:
            self._parameters = toml.load(f)

        # Setup logger
        setup_logger(self._parameters)

        # Parse user-inputs
        tams_subdict = self._parameters["montecarlo"]
        if "ntrajectories" not in tams_subdict:
            err_msg = "TAMS 'ntrajectories' must be specified in the input file !"
            _logger.exception(err_msg)
            raise ValueError

        self._plot_diags = tams_subdict.get("plot_diagnostics", False)


    def n_traj(self) -> int:
        """Return the number of trajectory used for TAMS.

        Note that this is the requested number of trajectory, not
        the current length of the trajectory ensemble.

        Return:
            number of trajectory
        """
        return self._tdb.n_traj()

    def generate_trajectory_ensemble(self) -> None:
        """Schedule the generation of an ensemble of stochastic trajectories.

        Loop over all the trajectories in the database and schedule
        advancing them to either end time or convergence with the
        runner.

        The runner will use the number of workers specified in the
        input file under the runner section.

        Raises:
            Error if the runner fails
        """
        inf_msg = f"Creating a Monte Carlo ensemble of {self._tdb.n_traj()} trajectories"
        _logger.info(inf_msg)

        with get_runner_type(self._parameters)(
            self._parameters, pool_worker, self._parameters.get("runner", {}).get("nworker_init", 1)
        ) as runner:
            for t in self._tdb.traj_list():
                task = [t, self._end_date, self._tdb.pool_file(), self._tdb.path()]
                runner.make_promise(task)

            try:
                t_list = runner.execute_promises()
            except:
                err_msg = f"Failed to generate the ensemble of {self._tdb.n_traj()} trajectories"
                _logger.exception(err_msg)
                raise

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        self._tdb.update_traj_list(t_list)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def compute_probability(self) -> float:
        """Compute the probability using MonteCarlo.

        Returns:
            the transition probability
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using MonteCarlo"
        _logger.info(inf_msg)

        # Generate the initial trajectory ensemble
        self.generate_trajectory_ensemble()

        # Get the transition probability
        transition_probability = self._tdb.count_converged_traj() / self._tdb.n_traj()

        self._tdb.info()

        return transition_probability

    def execute_sampling(self,
                         database: Database) -> None:
        """Shallow wrapper to enable sampler."""
        self._tdb = database
        self._tdb.load_data()

        # Initialize an empty trajectory ensemble
        if self._tdb.is_empty():
            self._tdb.init_active_ensemble()

        self.compute_probability()

    def initialize_db(self) -> type[Database]:
        """Return an initialized database of the TAMS sampling strategy."""
        return Database(fmodel_t=self._fmodel_t,
                        params=self._parameters,
                        ntraj=self._parameters["montecarlo"]["ntrajectories"])

    def get_database(self) -> Database:
        """Accessor to database.

        Returns:
            A reference to the database in use
        """
        return self._tdb

    def __del__(self) -> None:
        """Destructor.

        It is mostly useful on Windows systems
        """
        # Force deletion of database
        if hasattr(self, "_tdb"):
            del self._tdb
