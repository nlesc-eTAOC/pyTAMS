"""The GiardinaKurchanLecomteTailleur sampling strategy."""

import logging
from math import isclose
from pathlib import Path
from typing import Any
from pyrevs.core import Config
from pyrevs.core import RuntimeConfig
from pyrevs.database import Database
from pyrevs.database import DatabaseCoreSpec
from pyrevs.runner import BaseRunner
from pyrevs.runner import RunnerConfig
from pyrevs.runner import make_runner
from pyrevs.runner import pool_worker
from pyrevs.strategies.base import BaseSamplingStrategy
from pyrevs.strategies.base import TerminationCriterion
from pyrevs.strategies.base import TimeInterruptionCriterion
from pyrevs.strategies.base import TimeTerminationCriterion
from .config import GKTLConfig
from .extension import GKTLDatabaseExtension

_logger = logging.getLogger(__name__)


@BaseSamplingStrategy.register("gktl")
class GKTL(BaseSamplingStrategy):
    """A strategy class implementing GKTL.

    Notes:
        This strategy relies on time management provided by
        BaseSamplingStrategy (e.g. ``self._end_date``, ``elapsed_time()``).
    """

    def __init__(
        self,
        fmodel_t: Any,
        runtime_cfg: RuntimeConfig,
        runner_cfg: RunnerConfig,
        strategy_cfg: GKTLConfig,
        deterministic: bool,
    ) -> None:
        """Initialize a GKTL object.

        Args:
            fmodel_t: the forward model type
            runtime_cfg: the runtime config
            runner_cfg: the runner config
            strategy_cfg: the gktl config
            deterministic: the deterministic flag

        Raises:
            ValueError: if necessary config parameters are not found
        """
        self._fmodel_t = fmodel_t
        self._gktl_cfg = strategy_cfg
        self._gktl_cfg.validate()
        self._runner_cfg = runner_cfg
        self._loglevel = runtime_cfg.loglevel
        self._logfile = runtime_cfg.logfile
        self._deterministic = deterministic
        self._term_crit: list[TerminationCriterion] = []

        if strategy_cfg.end_time is not None:
            self._term_crit.append(TimeTerminationCriterion(strategy_cfg.end_time))

    def generate_trajectory_ensemble(self, database: Database, plot_diags: bool) -> None:
        """Schedule the generation of an ensemble of stochastic trajectories.

        Loop over all the trajectories in the database and schedule
        advancing them to either end time or convergence with the
        runner.

        The runner will use the number of workers specified in the
        input file under the runner section.

        Args:
            database: the sampling database
            plot_diags: whether to plot diagnostics

        Raises:
            Error if the runner fails
        """
        inf_msg = f"Creating a biased GKTL ensemble of {database.n_traj()} trajectories"
        _logger.info(inf_msg)

        with make_runner(
            self._runner_cfg,
            pool_worker,
            loglevel=self._loglevel,
            logfile=self._logfile,
            max_workers=self._gktl_cfg.ntrajectories,
        ) as runner:
            # Perform the initial step
            # up to the first resampling interval
            inf_msg = "GKTL step 0 - Starting at time: 0.0"
            _logger.info(inf_msg)

            # The sampling end time of the trajectories is controlled
            # through the termination criterion
            self._term_crit.append(TimeInterruptionCriterion(self._gktl_cfg.resampling_interval))
            advanced_time = self._one_gktl_step(database, runner, 0, plot_diags)

            # Initialize cummulative time
            gktl_time = advanced_time

            # Initialize the resampling counter
            k = 1

            # Iterate until all converged or end time is reached
            while not database.all_converged() and not (isclose(gktl_time, self._gktl_cfg.end_time, abs_tol=1e-9) or
                    gktl_time >= self._gktl_cfg.end_time):

                # Loggign info
                inf_msg = f"GKTL step {k:5} - Starting at time: {gktl_time}"
                _logger.info(inf_msg)

                # Update the sampling interval termination criterion
                self._term_crit[-1] = TimeInterruptionCriterion(
                    gktl_time + self._gktl_cfg.resampling_interval
                )

                # Perform a step
                advanced_time = self._one_gktl_step(database, runner, k, plot_diags)

                # Update cummulative time
                gktl_time += advanced_time

                # Update resampling counter
                k += 1

        if plot_diags:
            pltfile = f"Score_GKTL_{k:06}.png"
            if Path(pltfile).exists():
                wrn_msg = f"Attempting to overwrite the plot file {pltfile}"
                _logger.warning(wrn_msg)
            database.plot_score_functions(pltfile)


        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def _one_gktl_step(self,
                       database: Database,
                       runner: BaseRunner,
                       k: int,
                       plot_diags: bool) -> float:
        """Perform one step of the GKTL algorithm.

        Advance all the trajectories to the next resampling interval
        of the end time.

        Args:
            database: the trajectory database
            runner: the runner
            k: the resampling interval counter
            plot_diags: whether to plot diagnostics

        Returns:
            the time effectively advanced by the trajectories
        """
        # Plot trajectory database scores
        if plot_diags and k > 0:
            pltfile = f"Score_GKTL_{k:06}.png"
            if Path(pltfile).exists():
                wrn_msg = f"Attempting to overwrite the plot file {pltfile}"
                _logger.warning(wrn_msg)
            database.plot_score_functions(pltfile)

        # Assume that all the trajectories have advanced to the same
        # time already. FP uncertainty is not taken into account.
        traj_start_time = database.traj_list()[0].current_time()

        for t in database.traj_list():
            task = [t, self._term_crit, self._end_date, database.pool_file(), database.path()]
            runner.make_promise(task)

        try:
            t_list = runner.execute_promises()
        except Exception as exc:
            err_msg = f"Failed to generate the ensemble of {database.n_traj()} trajectories"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg) from exc

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        database.update_traj_list(t_list)

        # The trajectory advanced with its own step size. If
        # the resampling interval is not a multiple of the
        # step size, the actual end of the trajectory will
        # be not match the requested end of the resampling interval.
        traj_final_time = database.traj_list()[0].current_time()

        return traj_final_time - traj_start_time

    def compute_probability(self, database: Database, plot_diags: bool) -> float:
        """Compute the rare-event probability using MonteCarlo.

        Args:
            database: the trajectory database
            plot_diags: whether to plot diagnostics

        Returns:
            the rare-event probability
        """
        # Generate the initial trajectory ensemble
        self.generate_trajectory_ensemble(database, plot_diags)

        return database.get_event_probability()

    def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Shallow wrapper to enable sampler."""
        database.load_data()

        # Initialize an empty trajectory ensemble
        database.init_active_ensemble()

        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using MonteCarlo"
        _logger.info(inf_msg)

        proba = self.compute_probability(database, plot_diags)


        database.info()
        inf_msg = f"Event probability: {proba}"
        _logger.info(inf_msg)

    def initialize_database_schema(self, database: Database, diag_configs: dict[str, Config] | None) -> None:
        """Initialize database core state."""
        spec = DatabaseCoreSpec(
            ntraj=self._gktl_cfg.ntrajectories,
            strategy="gktl",
            deterministic=self._deterministic,
            diag_configs=diag_configs,
        )
        database.initialize_core_state(spec)

        # Setup GKTL extension
        self._db_ext = GKTLDatabaseExtension()
        self._db_ext.initialize(database)
        database.attach_extension(self._db_ext)
