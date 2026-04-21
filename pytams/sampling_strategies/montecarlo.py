"""The MonteCarlo sampling strategy."""

import logging
from typing import Any
from pytams.base_strategy import BaseSamplingStrategy
from pytams.database import Database
from pytams.taskrunner import get_runner_type
from pytams.worker import pool_worker

_logger = logging.getLogger(__name__)


@BaseSamplingStrategy.register("montecarlo")
class MonteCarlo(BaseSamplingStrategy):
    """A strategy class implementing MonteCarlo.

    Monte-Carlo or Direct Numerical Simulation (DNS) is not per-se
    a sampling strategy tailored for rare events but it provides
    a baseline for comparison with other sampling strategies.

    An ensemble of size n_traj is constructed and the rare-event probability
    is simply computed as the ratio of the number of converged trajectories
    to the total number of trajectories in the ensemble n_traj.

    In practice, this is the first step of a TAMS or AMS run (depending
    on the termination condition), such that this class is a lightweight
    version of these other strategies.
    """

    def __init__(self, fmodel_t: Any, parameters: dict[Any, Any]) -> None:
        """Initialize a Monte-Carlo object.

        Args:
            fmodel_t: the forward model type
            parameters: a dictionary of parameters

        Raises:
            ValueError: if necessary parameters are not found
        """
        self._fmodel_t = fmodel_t
        self._parameters = parameters

        # Parse user-inputs
        mc_subdict = self._parameters["montecarlo"]
        if "ntrajectories" not in mc_subdict:
            err_msg = "Monte-Carlo 'ntrajectories' must be specified in the input file !"
            _logger.exception(err_msg)
            raise ValueError

    def generate_trajectory_ensemble(self, tdb: Database) -> None:
        """Schedule the generation of an ensemble of stochastic trajectories.

        Loop over all the trajectories in the database and schedule
        advancing them to either end time or convergence with the
        runner.

        The runner will use the number of workers specified in the
        input file under the runner section.

        Raises:
            Error if the runner fails
        """
        inf_msg = f"Creating a Monte Carlo ensemble of {tdb.n_traj()} trajectories"
        _logger.info(inf_msg)

        with get_runner_type(self._parameters)(
            self._parameters, pool_worker, self._parameters.get("runner", {}).get("nworker_init", 1)
        ) as runner:
            for t in tdb.traj_list():
                task = [t, self._end_date, tdb.pool_file(), tdb.path()]
                runner.make_promise(task)

            try:
                t_list = runner.execute_promises()
            except:
                err_msg = f"Failed to generate the ensemble of {tdb.n_traj()} trajectories"
                _logger.exception(err_msg)
                raise

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        tdb.update_traj_list(t_list)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def compute_probability(self, tdb: Database) -> float:
        """Compute the rare-event probability using MonteCarlo.

        Returns:
            the transition probability
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using MonteCarlo"
        _logger.info(inf_msg)

        # Generate the initial trajectory ensemble
        self.generate_trajectory_ensemble(tdb)

        # Get the transition probability
        transition_probability = tdb.count_converged_traj() / tdb.n_traj()

        tdb.info()

        return transition_probability

    def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Shallow wrapper to enable sampler."""
        database.load_data()

        self._plot_diags = plot_diags

        # Initialize an empty trajectory ensemble
        if database.is_empty():
            database.init_active_ensemble()

        self.compute_probability(database)

    def initialize_db(self) -> Database:
        """Return an initialized database of the Monte-Carlo sampling strategy."""
        return Database(
            fmodel_t=self._fmodel_t,
            params=self._parameters,
            strategy="montecarlo",
            ntraj=self._parameters["montecarlo"]["ntrajectories"],
            read_only=False,
        )
