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
    1. Initialization of the trajectory pool
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
        _init_pool_only: whether or not to stop after initializing the trajectory pool
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
        self._init_pool_only = tams_subdict.get("pool_only", False)

        # Database
        self._tdb = Database(fmodel_t, self._parameters, n_traj, n_split_iter)
        self._tdb.load_data()

        # Time management uses UTC date
        # to make sure workers are always in sync
        self._startDate: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        self._endDate: datetime.datetime = self._startDate + datetime.timedelta(seconds=self._wallTime)

        # Initialize trajectory pool
        if self._tdb.is_empty():
            self._tdb.init_pool()

    def n_traj(self) -> int:
        """Return the number of trajectory used for TAMS.

        Note that this is the requested number of trajectory, not
        the current length of the trajectory pool.

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

    def generate_trajectory_pool(self) -> None:
        """Schedule the generation of a pool of stochastic trajectories.

        Loop over all the trajectories in the database and schedule
        advancing them to either end time or convergence with the
        runner.

        The runner will use the number of workers specified in the
        input file under the runner section.

        Raises:
            Error if the runner fails
        """
        inf_msg = f"Creating the initial pool of {self._tdb.n_traj()} trajectories"
        _logger.info(inf_msg)

        with get_runner_type(self._parameters)(
            self._parameters, pool_worker, self._parameters.get("runner", {}).get("nworker_init", 1)
        ) as runner:
            for t in self._tdb.traj_list():
                task = [t, self._endDate, self._tdb.path()]
                runner.make_promise(task)

            try:
                t_list = runner.execute_promises()
            except:
                err_msg = f"Failed to generate the initial pool of {self._tdb.n_traj()} trajectories"
                _logger.exception(err_msg)
                raise

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        self._tdb.update_traj_list(t_list)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def check_exit_splitting_loop(self, k: int) -> tuple[bool, npt.NDArray[np.number]]:
        """Check for exit criterion of the splitting loop.

        Args:
            k: loop counter

        Returns:
            bool to trigger splitting loop break
            array of maximas accros all trajectories
        """
        # Check for walltime
        if self.out_of_time():
            warn_msg = f"Ran out of time after {k} splitting iterations"
            _logger.warning(warn_msg)
            return True, np.empty(1)

        # Gather max score from all trajectories
        # and check for early convergence
        all_converged = True
        maxes = np.zeros(self._tdb.traj_list_len())
        for i in range(self._tdb.traj_list_len()):
            maxes[i] = self._tdb.get_traj(i).score_max()
            all_converged = all_converged and self._tdb.get_traj(i).is_converged()

        # Exit if our work is done
        if all_converged:
            inf_msg = f"All trajectories converged after {k} splitting iterations"
            _logger.info(inf_msg)
            return True, np.empty(1)

        # Exit if splitting is stalled
        if (np.amax(maxes) - np.amin(maxes)) < STALL_TOL:
            err_msg = f"Splitting is stalling with all trajectories stuck at a score_max: {np.amax(maxes)}"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        return False, maxes

    def finish_ongoing_splitting(self) -> None:
        """Check and finish unfinished splitting iterations.

        If the run was interupted during a splitting iteration,
        the branched trajectory might not have ended yet. In that case,
        a list of trajectories to finish is listed in the database.
        """
        # Check the database for unfinished splitting iteration when restarting.
        # At this point, branching has been done, but advancing to final
        # time is still ongoing.
        ongoing_list = self._tdb.get_ongoing()
        if ongoing_list:
            inf_msg = f"Unfinished splitting iteration detected, traj {self._tdb.get_ongoing()} need(s) finishing"
            _logger.info(inf_msg)
            with get_runner_type(self._parameters)(
                self._parameters, pool_worker, self._parameters.get("runner", {}).get("nworker_iter", 1)
            ) as runner:
                n_branch = len(ongoing_list)
                for i in ongoing_list:
                    t = self._tdb.get_traj(i)
                    task = [t, self._endDate, self._tdb.path()]
                    runner.make_promise(task)
                finished_traj = runner.execute_promises()

                for t in finished_traj:
                    self._tdb.overwrite_traj(t.id(), t)

                # Clear list of ongoing branches
                self._tdb.reset_ongoing()

                # Increment splitting index
                k = self._tdb.k_split() + n_branch
                self._tdb.set_k_split(k)
                self._tdb.save_splitting_data()

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
        """Schedule splitting of the initial pool of stochastic trajectories.

        Perform the multi-level splitting iterations, possibly restarting multiple
        trajectories at each iterations. All the trajectories in an iterations are
        advanced together, such each iteration takes the maximum duration among
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

                # Check for early exit conditions
                early_exit, maxes = self.check_exit_splitting_loop(k)
                if early_exit:
                    break

                self._tdb.append_minmax(k, np.min(maxes), np.max(maxes))

                # Plot trajectory database scores
                if self._plot_diags:
                    pltfile = f"Score_k{k:05}.png"
                    self._tdb.plot_score_functions(pltfile)

                # Get the nworker lower scored trajectories
                # or more if equal score
                min_idx_list, min_vals = get_min_scored(maxes, runner.n_workers())

                # Randomly select trajectory to branch from
                rest_idx = self.get_restart_at_random(min_idx_list)
                n_branch = len(min_idx_list)

                self._tdb.append_bias(n_branch)
                self._tdb.append_weight(self._tdb.weights()[-1] * (1 - self._tdb.biases()[-1] / self._tdb.n_traj()))

                self._tdb.append_splitting_data(
                    k, n_branch, min_idx_list, rest_idx, min_vals.tolist(), [np.min(maxes), np.max(maxes)]
                )

                # Assemble a list of promises
                for i in range(n_branch):
                    task = [
                        self._tdb.get_traj(rest_idx[i]),
                        self._tdb.get_traj(min_idx_list[i]),
                        min_vals[i],
                        self._endDate,
                        self._tdb.path(),
                    ]
                    runner.make_promise(task)

                try:
                    restarted_trajs = runner.execute_promises()
                except Exception:
                    err_msg = f"Failed to branch {n_branch} trajectories at iteration {k}"
                    _logger.exception(err_msg)
                    self._tdb.save_splitting_data(min_idx_list)
                    raise

                # Update the trajectory database
                for t in restarted_trajs:
                    self._tdb.overwrite_traj(t.id(), t)

                if self.out_of_time():
                    # Save splitting data with ongoing trajectories
                    # but do not increment splitting index yet
                    self._tdb.save_splitting_data(min_idx_list)
                    warn_msg = f"Ran out of time after {k} splitting iterations"
                    _logger.warning(warn_msg)
                    break

                # Update the trajectory database, increment splitting index
                k = k + n_branch
                self._tdb.set_k_split(k)
                self._tdb.save_splitting_data()

    def compute_probability(self) -> float:
        """Compute the probability using TAMS.

        Returns:
            the transition probability
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using TAMS"
        _logger.info(inf_msg)

        # Skip pool stage if splitting iterative
        # process has started
        skip_pool = self._tdb.k_split() > 0

        # Generate the initial trajectory pool
        if not skip_pool:
            self.generate_trajectory_pool()

        # Check for early convergence
        all_converged = True
        for t in self._tdb.traj_list():
            if not t.is_converged():
                all_converged = False
                break

        if not skip_pool and all_converged:
            inf_msg = "All trajectories converged prior to splitting !"
            _logger.info(inf_msg)
            return 1.0

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        if self._init_pool_only:
            warn_msg = "Stopping after the pool stage !"
            _logger.warning(warn_msg)
            return -1.0

        # Perform multilevel splitting
        if not all_converged:
            self.do_multilevel_splitting()

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        trans_prob = self._tdb.get_transition_probability()

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

        return trans_prob
