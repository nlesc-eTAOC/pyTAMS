"""The (T)AMS sampling strategy."""

import logging
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from .config import AMSConfig
from pytams.config import Config
from pytams.config import RuntimeConfig
from pytams.database import Database
from pytams.runner import make_runner
from pytams.runner import ms_worker
from pytams.runner import pool_worker
from pytams.runner import RunnerConfig
from pytams.strategies.base_strategy import BaseSamplingStrategy
from pytams.utils import get_min_scored

_logger = logging.getLogger(__name__)

STALL_TOL = 1e-10


@BaseSamplingStrategy.register("ams")
class AMS(BaseSamplingStrategy):
    """A strategy class implementing (T)AMS.

    The interface to (T)AMS, implementing the main steps of
    the algorithm.

    Initialization of the (T)AMS class requires a forward model
    type which encapsulate all the model-specific code, and
    an optional list of options.

    The algorithm is roughly divided in two steps:
    1. Initialization of the trajectory ensemble
    2. Splitting iterations

    Separate control of the parallelism is provided for
    both steps.

    All the algorithm data are contained in the (T)AMS database.
    For control purposes, a walltime limit is also provided. It is
    passed to working and lead to the termination of the algorithm
    in a state that can be saved to disk and restarted at a later stage.

    Attributes:
        _fmodel_t: the forward model type
        _config: the config object
        _wallTime: the walltime limit
        _init_ensemble_only: whether or not to stop after initializing the trajectory ensemble
    """

    def __init__(self, fmodel_t: Any, config: Config, runtime_cfg: RuntimeConfig, deterministic: bool) -> None:
        """Initialize a AMS strategy.

        Args:
            fmodel_t: the forward model type
            config: the config object
            runtime_cfg: the runtime config dataclass
            deterministic: the deterministic flag trigger deterministic runs

        Raises:
            ValueError: if necessary parameters are not found
        """
        self._fmodel_t = fmodel_t
        self._config = config
        self._ams_cfg = config.load(AMSConfig)
        self._ams_cfg.validate()
        self._runner_cfg = config.load(RunnerConfig)
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

        Args:
            tdb: the AMS database

        Raises:
            Error if the runner fails
        """
        inf_msg = f"Creating the initial ensemble of {tdb.n_traj()} trajectories"
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
                err_msg = f"Failed to generate the initial ensemble of {tdb.n_traj()} trajectories"
                _logger.exception(err_msg)
                raise RuntimeError(err_msg) from exc

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        tdb.update_traj_list(t_list)

        if tdb.count_terminated_traj() == tdb.n_traj():
            tdb.set_init_ensemble_flag(True)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def check_exit_splitting_loop(self, tdb: Database, k: int) -> tuple[bool, npt.NDArray[np.number]]:
        """Check for exit criterion of the splitting loop.

        Args:
            tdb: the AMS database
            k: loop counter

        Returns:
            bool to trigger splitting loop break
            array of maximas across all trajectories
        """
        # Gather max score from all trajectories
        # and check for early convergence
        all_converged = True
        maxes = np.zeros(tdb.traj_list_len(), dtype=float)
        for i in range(tdb.traj_list_len()):
            maxes[i] = tdb.get_traj(i).score_max()
            all_converged = all_converged and tdb.get_traj(i).is_converged()

        # Check for walltime
        if self.out_of_time():
            warn_msg = f"Ran out of time after {k} splitting iterations"
            _logger.warning(warn_msg)
            return True, maxes

        # Exit if our work is done
        if all_converged:
            inf_msg = f"All trajectories converged after {k} splitting iterations"
            _logger.info(inf_msg)
            return True, maxes

        # Exit if splitting is stalled
        if (np.amax(maxes) - np.amin(maxes)) < STALL_TOL:
            err_msg = f"Splitting is stalling with all trajectories stuck at a score_max: {np.amax(maxes)}"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        return False, maxes

    def finish_ongoing_splitting(self, tdb: Database) -> None:
        """Check and finish unfinished splitting iterations.

        If the run was interrupted during a splitting iteration,
        the branched trajectories might not have terminated yet. In that case,
        a list of trajectories to finish is listed in the database.
        """
        # Check the database for unfinished splitting iteration when restarting.
        # At this point, branching has been done, but advancing to final
        # time is still ongoing.
        ongoing_list = tdb.get_ongoing()
        if ongoing_list:
            inf_msg = f"Unfinished splitting iteration detected, traj {ongoing_list} need(s) finishing"
            _logger.info(inf_msg)
            with make_runner(
                self._runner_cfg,
                pool_worker,
                is_pool_worker=False,
                loglevel=self._loglevel,
                logfile=self._logfile,
            ) as runner:
                for i in ongoing_list:
                    t = tdb.get_traj(i)
                    task = [t, self._end_date, tdb.pool_file(), tdb.path()]
                    runner.make_promise(task)

                try:
                    finished_traj = runner.execute_promises()
                except Exception as exc:
                    err_msg = f"Failed to finish branching {len(ongoing_list)} trajectories"
                    _logger.exception(err_msg)
                    raise RuntimeError(err_msg) from exc

                _logger.info("Done with unfinished")

                for t in finished_traj:
                    tdb.overwrite_traj(t.id(), t)

                # Wrap up the iteration by updating its status in the
                # database and incrementing the iteration counter
                tdb.mark_last_splitting_iteration_as_done()

    def get_restart_at_random(self, tdb: Database, min_idx_list: list[int]) -> list[int]:
        """Get a list of trajectory index to restart from at random.

        Select trajectories to restart from among the ones not
        in min_idx_list.

        Args:
            tdb: the AMS database
            min_idx_list: list of trajectory index to restart from

        Returns:
            list of trajectory index to restart from
        """
        # Enable deterministic runs by setting a (different) seed
        # for each splitting iteration
        rng = np.random.default_rng(seed=42 * tdb.k_split()) if self._deterministic else np.random.default_rng()
        rest_idx = [-1] * len(min_idx_list)
        for i in range(len(min_idx_list)):
            rest_idx[i] = min_idx_list[0]
            while rest_idx[i] in min_idx_list:
                rest_idx[i] = rng.integers(low=0, high=tdb.traj_list_len(), dtype=int)
        return rest_idx

    def do_multilevel_splitting(self, tdb: Database, plot_diags: bool) -> None:
        """Schedule splitting of the initial ensemble of stochastic trajectories.

        Perform the multi-level splitting iterations, possibly restarting multiple
        trajectories at each iterations. All the trajectories in an iterations are
        advanced together, such that each iteration takes the maximum duration among
        the branched trajectories.

        If the walltime is exceeded, the splitting loop is stopped and ongoing
        trajectories are flagged in the database in order to finish them upon
        restart.

        The runner will use the number of workers specified in the
        input file under the runner section.

        Args:
            tdb: the AMS database
            plot_diags: whether or not to plot diagnostics

        Raises:
            Error if the runner fails
        """
        inf_msg = "Using multi-level splitting to get the probability"
        _logger.info(inf_msg)

        # Finish any unfinished splitting iteration
        self.finish_ongoing_splitting(tdb)

        # Initialize splitting iterations counter
        k = tdb.k_split()

        with make_runner(
            self._runner_cfg,
            ms_worker,
            is_pool_worker=False,
            loglevel=self._loglevel,
            logfile=self._logfile,
        ) as runner:
            while k < tdb.n_split_iter():
                inf_msg = f"Starting AMS iter. {k} with {runner.n_workers()} workers"
                _logger.info(inf_msg)

                # Plot trajectory database scores
                if plot_diags:
                    pltfile = f"Score_k{k:05}.png"
                    if Path(pltfile).exists():
                        wrn_msg = f"Attempting to overwrite the plot file {pltfile}"
                        _logger.warning(wrn_msg)
                    tdb.plot_score_functions(pltfile)

                # Get the ensemble maximums and check for early exit conditions
                early_exit, maxes = self.check_exit_splitting_loop(tdb, k)

                # Get the nworker lower scored trajectories
                # or more if equal score
                min_idx_list, min_vals = get_min_scored(maxes, runner.n_workers())

                # Randomly select trajectory to branch from
                ancestor_idx = self.get_restart_at_random(tdb, min_idx_list)
                n_branch = len(min_idx_list)

                # Update the database with the data of the current
                # iteration
                tdb.append_splitting_iteration_data(
                    k, n_branch, min_idx_list, ancestor_idx, min_vals.tolist(), [np.min(maxes), np.max(maxes)]
                )

                # Query the current iteration weight
                # to compute the individual weight of each trajectory in the ensemble
                # at the end of the splitting iteration
                new_traj_weight = tdb.weights()[-1]

                # Exit the loop if needed
                if early_exit:
                    # If AMS converged, final update of the weights.
                    if tdb.all_converged():
                        tdb.update_trajectories_weights()
                    break

                # Assemble a list of promises
                # and archive the discarded trajectories
                for i in range(n_branch):
                    # Archive
                    tdb.archive_trajectory(tdb.get_traj(min_idx_list[i]))

                    # Worker task
                    task = [
                        tdb.get_traj(ancestor_idx[i]),
                        tdb.get_traj(min_idx_list[i]),
                        np.max(min_vals),
                        new_traj_weight,
                        self._end_date,
                        tdb.pool_file(),
                        tdb.path(),
                    ]
                    runner.make_promise(task)

                try:
                    restarted_trajs = runner.execute_promises()
                except Exception as exc:
                    err_msg = f"Failed to branch {n_branch} trajectories at iteration {k}"
                    _logger.exception(err_msg)
                    raise RuntimeError(err_msg) from exc

                # Update the trajectories in the database
                for t in restarted_trajs:
                    tdb.overwrite_traj(t.id(), t)

                # Update the weights of all trajectories in the ensemble with the current
                # iteration weight
                tdb.update_trajectories_weights()

                if self.out_of_time():
                    # Save splitting data with ongoing trajectories
                    # but do not increment splitting index yet
                    warn_msg = f"Ran out of time after {k} splitting iterations"
                    _logger.warning(warn_msg)
                    break

                # Wrap up the iteration by updating its status in the
                # database and incrementing the iteration counter
                tdb.mark_last_splitting_iteration_as_done()
                k = k + n_branch

    def compute_probability(self, tdb: Database, plot_diags: bool) -> float:
        """Compute the probability using AMS.

        Args:
            tdb: the AMS database
            plot_diags: whether or not to plot diagnostics

        Returns:
            the transition probability
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using AMS"
        _logger.info(inf_msg)

        # Generate the initial trajectory ensemble
        init_ensemble_need_work = not tdb.init_ensemble_done()
        if init_ensemble_need_work:
            self.generate_trajectory_ensemble(tdb)

        # Check for early convergence
        all_converged = True
        for t in tdb.traj_list():
            if not t.is_converged():
                all_converged = False
                break

        if init_ensemble_need_work and all_converged:
            inf_msg = "All trajectories in the ensemble converged prior to splitting !"
            _logger.info(inf_msg)
            return 1.0

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        if self._ams_cfg.init_ensemble_only:
            warn_msg = "Stopping after the initial ensemble stage !"
            _logger.warning(warn_msg)
            return -1.0

        # Perform multilevel splitting
        if not all_converged:
            self.do_multilevel_splitting(tdb, plot_diags)

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        transition_probability = tdb.get_transition_probability()

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

        # Load the archived trajectories data since the workers
        # discarded them but the persistent Python process did not
        # kept track
        tdb.load_archived_trajectories()

        tdb.info()

        return transition_probability

    def _execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Shallow wrapper to enable sampler."""
        database.load_data()

        # Initialize the active ensemble
        database.init_active_ensemble()

        self.compute_probability(database, plot_diags)

    def initialize_db(self, diag_configs: dict[str, Config] | None) -> Database:
        """Return an initialized database of the AMS sampling strategy."""
        return Database(
            fmodel_t=self._fmodel_t,
            config=self._config,
            diag_configs=diag_configs,
            strategy="ams",
            ntraj=self._ams_cfg.ntrajectories,
            nsplititer=self._ams_cfg.nsplititer,
            read_only=False,
        )
