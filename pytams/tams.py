"""The main TAMS class."""

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import toml
from pytams.database import Database
from pytams.taskrunner import get_runner_type
from pytams.utils import get_min_scored
from pytams.utils import setup_logger
from pytams.worker import ms_worker
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


class TAMS:
    """A class implementing TAMS.

    The interface to TAMS, implementing the main steps of
    the algorithm.

    Initialization of the TAMS class requires a forward model
    type which encapsulate all the model-specific code, and
    an optional list of options.

    The algorithm is roughly divided in two steps:
    1. Initialization of the trajectory ensemble
    2. Splitting iterations

    Separate control of the parallelism is provided for
    both steps.

    All the algorithm data are contained in the TAMS database.
    For control purposes, a walltime limit is also provided. It is
    passed to working and lead to the termination of the algorithm
    in a state that can be saved to disk and restarted at a later stage.

    Attributes:
        _fmodel_t: the forward model type
        _parameters: the dictionary of parameters
        _wallTime: the walltime limit
        _startDate: the date the algorithm started
        _plot_diags: whether or not to plot diagnostics during splitting iterations
        _init_ensemble_only: whether or not to stop after initializing the trajectory ensemble
        _tdb: the trajectory database (containing all trajectories)
    """

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
        tams_subdict = self._parameters["tams"]
        if "ntrajectories" not in tams_subdict or "nsplititer" not in tams_subdict:
            err_msg = "TAMS 'ntrajectories' and 'nsplititer' must be specified in the input file !"
            _logger.exception(err_msg)
            raise ValueError

        n_traj: int = tams_subdict.get("ntrajectories")
        n_split_iter: int = tams_subdict.get("nsplititer")
        self._wallTime: float = tams_subdict.get("walltime", 24.0 * 3600.0)
        self._plot_diags = tams_subdict.get("diagnostics", False)
        self._init_ensemble_only = tams_subdict.get("init_ensemble_only", False)

        # Database
        self._tdb = Database(fmodel_t, self._parameters, n_traj, n_split_iter, read_only=False)
        self._tdb.load_data()

        # Time management uses UTC date
        # to make sure workers are always in sync
        self._startDate: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        self._endDate: datetime.datetime = self._startDate + datetime.timedelta(seconds=self._wallTime)

        # Initialize an empty trajectory ensemble
        if self._tdb.is_empty():
            self._tdb.init_active_ensemble()

    def n_traj(self) -> int:
        """Return the number of trajectory used for TAMS.

        Note that this is the requested number of trajectory, not
        the current length of the trajectory ensemble.

        Return:
            number of trajectory
        """
        return self._tdb.n_traj()

    def elapsed_time(self) -> float:
        """Return the elapsed wallclock time.

        Since the initialization of the TAMS object [seconds].

        Returns:
           TAMS elapse time.

        Raises:
            ValueError: if the start date is not set
        """
        delta: datetime.timedelta = datetime.datetime.now(tz=datetime.timezone.utc) - self._startDate
        if delta:
            return delta.total_seconds()

        err_msg = "TAMS start date is not set !"
        _logger.exception(err_msg)
        raise ValueError

    def remaining_walltime(self) -> float:
        """Return the remaining wallclock time.

        [seconds]

        Returns:
           TAMS remaining wall time.
        """
        return self._wallTime - self.elapsed_time()

    def out_of_time(self) -> bool:
        """Return true if insufficient walltime remains.

        Allows for 5% slack to allows time for workers to finish
        their work (especially with Dask+Slurm backend).

        Returns:
           boolean indicating wall time availability.
        """
        return self.remaining_walltime() < 0.05 * self._wallTime

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
        inf_msg = f"Creating the initial ensemble of {self._tdb.n_traj()} trajectories"
        _logger.info(inf_msg)

        with get_runner_type(self._parameters)(
            self._parameters, pool_worker, self._parameters.get("runner", {}).get("nworker_init", 1)
        ) as runner:
            for t in self._tdb.traj_list():
                task = [t, self._endDate, self._tdb.pool_file(), self._tdb.path()]
                runner.make_promise(task)

            try:
                t_list = runner.execute_promises()
            except:
                err_msg = f"Failed to generate the initial ensemble of {self._tdb.n_traj()} trajectories"
                _logger.exception(err_msg)
                raise

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        self._tdb.update_traj_list(t_list)

        if self._tdb.count_ended_traj() == self._tdb.n_traj():
            self._tdb.set_init_ensemble_flag(True)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def check_exit_splitting_loop(self, k: int) -> tuple[bool, npt.NDArray[np.number]]:
        """Check for exit criterion of the splitting loop.

        Args:
            k: loop counter

        Returns:
            bool to trigger splitting loop break
            array of maximas across all trajectories
        """
        # Gather max score from all trajectories
        # and check for early convergence
        all_converged = True
        maxes = np.zeros(self._tdb.traj_list_len())
        for i in range(self._tdb.traj_list_len()):
            maxes[i] = self._tdb.get_traj(i).score_max()
            all_converged = all_converged and self._tdb.get_traj(i).is_converged()

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

    def finish_ongoing_splitting(self) -> None:
        """Check and finish unfinished splitting iterations.

        If the run was interrupted during a splitting iteration,
        the branched trajectories might not have ended yet. In that case,
        a list of trajectories to finish is listed in the database.
        """
        # Check the database for unfinished splitting iteration when restarting.
        # At this point, branching has been done, but advancing to final
        # time is still ongoing.
        ongoing_list = self._tdb.get_ongoing()
        if ongoing_list:
            inf_msg = f"Unfinished splitting iteration detected, traj {ongoing_list} need(s) finishing"
            _logger.info(inf_msg)
            with get_runner_type(self._parameters)(
                self._parameters, pool_worker, self._parameters.get("runner", {}).get("nworker_iter", 1)
            ) as runner:
                for i in ongoing_list:
                    t = self._tdb.get_traj(i)
                    task = [t, self._endDate, self._tdb.pool_file(), self._tdb.path()]
                    runner.make_promise(task)

                try:
                    finished_traj = runner.execute_promises()
                except Exception:
                    err_msg = f"Failed to finish branching {len(ongoing_list)} trajectories"
                    _logger.exception(err_msg)
                    raise

                _logger.info("Done with unfinished")

                for t in finished_traj:
                    self._tdb.overwrite_traj(t.id(), t)

                # Wrap up the iteration by updating its status in the
                # database and incrementing the iteration counter
                self._tdb.mark_last_splitting_iteration_as_done()

    def get_restart_at_random(self, min_idx_list: list[int]) -> list[int]:
        """Get a list of trajectory index to restart from at random.

        Select trajectories to restart from among the ones not
        in min_idx_list.

        Args:
            min_idx_list: list of trajectory index to restart from

        Returns:
            list of trajectory index to restart from
        """
        # Enable deterministic runs by setting a (different) seed
        # for each splitting iteration
        if self._parameters.get("tams", {}).get("deterministic", False):
            rng = np.random.default_rng(seed=42 * self._tdb.k_split())
        else:
            rng = np.random.default_rng()
        rest_idx = [-1] * len(min_idx_list)
        for i in range(len(min_idx_list)):
            rest_idx[i] = min_idx_list[0]
            while rest_idx[i] in min_idx_list:
                rest_idx[i] = rng.integers(low=0, high=self._tdb.traj_list_len(), dtype=int)
        return rest_idx

    def do_multilevel_splitting(self) -> None:
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

        Raises:
            Error if the runner fails
        """
        inf_msg = "Using multi-level splitting to get the probability"
        _logger.info(inf_msg)

        # Finish any unfinished splitting iteration
        self.finish_ongoing_splitting()

        # Initialize splitting iterations counter
        k = self._tdb.k_split()

        with get_runner_type(self._parameters)(
            self._parameters, ms_worker, self._parameters.get("runner", {}).get("nworker_iter", 1)
        ) as runner:
            while k < self._tdb.n_split_iter():
                inf_msg = f"Starting TAMS iter. {k} with {runner.n_workers()} workers"
                _logger.info(inf_msg)

                # Plot trajectory database scores
                if self._plot_diags:
                    pltfile = f"Score_k{k:05}.png"
                    if Path(pltfile).exists():
                        wrn_msg = f"Attempting to overwrite the plot file {pltfile}"
                        _logger.warning(wrn_msg)
                    self._tdb.plot_score_functions(pltfile)

                # Get the ensemble maximums and check for early exit conditions
                early_exit, maxes = self.check_exit_splitting_loop(k)

                # Get the nworker lower scored trajectories
                # or more if equal score
                min_idx_list, min_vals = get_min_scored(maxes, runner.n_workers())

                # Randomly select trajectory to branch from
                ancestor_idx = self.get_restart_at_random(min_idx_list)
                n_branch = len(min_idx_list)

                # Update the database with the data of the current
                # iteration
                self._tdb.append_splitting_iteration_data(
                    k, n_branch, min_idx_list, ancestor_idx, min_vals.tolist(), [np.min(maxes), np.max(maxes)]
                )

                # Query the current iteration weight
                # to compute the individual weight of each trajectory in the ensemble
                # at the end of the splitting iteration
                new_traj_weight = self._tdb.weights()[-1] / float(self._tdb.n_traj())

                # Exit the loop if needed
                if early_exit:
                    break

                # Assemble a list of promises
                # and archive the discarded trajectories
                for i in range(n_branch):
                    # Archive
                    self._tdb.archive_trajectory(self._tdb.get_traj(min_idx_list[i]))

                    # Worker task
                    task = [
                        self._tdb.get_traj(ancestor_idx[i]),
                        self._tdb.get_traj(min_idx_list[i]),
                        min_vals[i],
                        new_traj_weight,
                        self._endDate,
                        self._tdb.pool_file(),
                        self._tdb.path(),
                    ]
                    runner.make_promise(task)

                try:
                    restarted_trajs = runner.execute_promises()
                except Exception:
                    err_msg = f"Failed to branch {n_branch} trajectories at iteration {k}"
                    _logger.exception(err_msg)
                    raise

                # Update the trajectories in the database
                for t in restarted_trajs:
                    self._tdb.overwrite_traj(t.id(), t)

                # Update the weights of all trajectories with the current
                # iteration weight
                self._tdb.update_trajectories_weights()

                if self.out_of_time():
                    # Save splitting data with ongoing trajectories
                    # but do not increment splitting index yet
                    warn_msg = f"Ran out of time after {k} splitting iterations"
                    _logger.warning(warn_msg)
                    break

                # Wrap up the iteration by updating its status in the
                # database and incrementing the iteration counter
                self._tdb.mark_last_splitting_iteration_as_done()
                k = k + n_branch

    def compute_probability(self) -> float:
        """Compute the probability using TAMS.

        Returns:
            the transition probability
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using TAMS"
        _logger.info(inf_msg)

        # Generate the initial trajectory ensemble
        init_ensemble_need_work = not self._tdb.init_ensemble_done()
        if init_ensemble_need_work:
            self.generate_trajectory_ensemble()

        # Check for early convergence
        all_converged = True
        for t in self._tdb.traj_list():
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

        if self._init_ensemble_only:
            warn_msg = "Stopping after the initial ensemble stage !"
            _logger.warning(warn_msg)
            return -1.0

        # Perform multilevel splitting
        if not all_converged:
            self.do_multilevel_splitting()

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        transition_probability = self._tdb.get_transition_probability()

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

        # Load the archived trajectories data since the workers
        # discarded them but the persistent Python process did not
        # kept track
        self._tdb.load_archived_trajectories()

        self._tdb.info()

        return transition_probability

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
