"""The MonteCarlo sampling strategy."""

import logging
from pathlib import Path
from typing import Any
from pytams.core import Config
from pytams.core import RuntimeConfig
from pytams.database import Database
from pytams.database import DatabaseCoreSpec
from pytams.runner import RunnerConfig
from pytams.runner import make_runner
from pytams.runner import pool_worker
from pytams.strategies.base import BaseSamplingStrategy
from .config import MCConfig
from .extension import MCDatabaseExtension

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

    Notes:
        This strategy relies on time management provided by
        BaseSamplingStrategy (e.g. ``self._end_date``, ``elapsed_time()``).
    """

    def __init__(
        self,
        fmodel_t: Any,
        runtime_cfg: RuntimeConfig,
        runner_cfg: RunnerConfig,
        strategy_cfg: MCConfig,
        deterministic: bool,
    ) -> None:
        """Initialize a Monte-Carlo object.

        Args:
            fmodel_t: the forward model type
            runtime_cfg: the runtime config
            runner_cfg: the runner config
            strategy_cfg: the montecarlo config
            deterministic: the deterministic flag

        Raises:
            ValueError: if necessary config parameters are not found
        """
        self._fmodel_t = fmodel_t
        self._mc_cfg = strategy_cfg
        self._mc_cfg.validate()
        self._runner_cfg = runner_cfg
        self._loglevel = runtime_cfg.loglevel
        self._logfile = runtime_cfg.logfile
        self._deterministic = deterministic

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

        with make_runner(
            self._runner_cfg,
            pool_worker,
            is_pool_worker=True,
            loglevel=self._loglevel,
            logfile=self._logfile,
        ) as runner:
            for t in tdb.traj_list():
                task = [t, self._end_date, tdb.pool_file(), tdb.path()]
                runner.make_promise(task)

            try:
                t_list = runner.execute_promises()
            except Exception as exc:
                err_msg = f"Failed to generate the ensemble of {tdb.n_traj()} trajectories"
                _logger.exception(err_msg)
                raise RuntimeError(err_msg) from exc

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        tdb.update_traj_list(t_list)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def compute_probability(self, tdb: Database) -> float:
        """Compute the rare-event probability using MonteCarlo.

        Returns:
            the rare-event probability
        """
        # Generate the initial trajectory ensemble
        self.generate_trajectory_ensemble(tdb)

        return tdb.get_event_probability()

    def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Shallow wrapper to enable sampler."""
        database.load_data()

        # Initialize an empty trajectory ensemble
        database.init_active_ensemble()

        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using MonteCarlo"
        _logger.info(inf_msg)

        proba = self.compute_probability(database)

        # Plot trajectory database scores
        if plot_diags:
            pltfile = "Score_MCEnd.png"
            if Path(pltfile).exists():
                wrn_msg = f"Attempting to overwrite the plot file {pltfile}"
                _logger.warning(wrn_msg)
            database.plot_score_functions(pltfile)

        database.info()
        inf_msg = f"Event probability: {proba}"
        _logger.info(inf_msg)

    def initialize_database_schema(self, database: Database, diag_configs: dict[str, Config] | None) -> None:
        """Initialize database core state."""
        spec = DatabaseCoreSpec(
            ntraj=self._mc_cfg.ntrajectories,
            strategy="montecarlo",
            deterministic=self._deterministic,
            diag_configs=diag_configs,
        )
        database.initialize_core_state(spec)

        # Setup MC extension
        self._db_ext = MCDatabaseExtension()
        self._db_ext.initialize(database)
        database.attach_extension(self._db_ext)
