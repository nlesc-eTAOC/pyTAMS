"""The main TAMS class."""
import argparse
import datetime
import logging
import os
from typing import Any
from typing import Optional
import numpy as np
import toml
from pytams.database import Database
from pytams.taskrunner import get_runner_type
from pytams.trajectory import Trajectory
from pytams.utils import get_min_scored
from pytams.utils import setup_logger
from pytams.worker import ms_worker
from pytams.worker import pool_worker

_logger = logging.getLogger(__name__)

class TAMSError(Exception):
    """Exception class for TAMS."""

    pass

def parse_cl_args(a_args: Optional[list[str]] = None) -> argparse.Namespace :
    """Parse provided list or default CL argv.

    Args:
        a_args: optional list of options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="pyTAMS input .toml file", default="input.toml")
    if a_args is not False:
        args = parser.parse_args(a_args)
    else:
        args = parser.parse_args()
    return args

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

    def __init__(self,
                 fmodel_t : Any,
                 a_args: Optional[list[str]] = None) -> None:
        """Initialize a TAMS run.

        Args:
            fmodel_t: the forward model type
            a_args: optional list of options

        Raises:
            TAMSError: if the input file is not found
        """
        self._fmodel_t = fmodel_t

        input_file = vars(parse_cl_args(a_args=a_args))["input"]
        if (not os.path.exists(input_file)):
            err_msg = f"Could not find the {input_file} TAMS input file !"
            _logger.error(err_msg)
            raise TAMSError(err_msg)

        with open(input_file, 'r') as f:
            self._parameters = toml.load(f)

        # Setup logger
        setup_logger(self._parameters)

        # Parse user-inputs
        tams_subdict = self._parameters["tams"]
        if ("ntrajectories" not in tams_subdict or
            "nsplititer" not in tams_subdict):
            err_msg = "TAMS 'ntrajectories' and 'nsplititer' must be specified in the input file !"
            _logger.error(err_msg)
            raise ValueError

        nTraj : int = tams_subdict.get("ntrajectories")
        nSplitIter : int = tams_subdict.get("nsplititer")
        self._wallTime : float = tams_subdict.get("walltime", 24.0*3600.0)
        self._plot_diags = tams_subdict.get("diagnostics", False)
        self._init_pool_only = tams_subdict.get("pool_only", False)

        # Database
        self._tdb = Database(fmodel_t,
                             self._parameters,
                             nTraj,
                             nSplitIter)

        # Time management uses UTC date
        # to make sure workers are always in sync
        self._startDate : datetime = datetime.datetime.utcnow()
        self._endDate : datetime = self._startDate + datetime.timedelta(seconds=self._wallTime)

        # Initialize trajectory pool
        if self._tdb.isEmpty():
            self.init_trajectory_pool()

    def nTraj(self) -> int:
        """Return the number of trajectory used for TAMS.

        Note that this is the requested number of trajectory, not
        the current length of the trajectory pool.

        Return:
            number of trajectory
        """
        return self._tdb.nTraj()

    def elapsed_time(self) -> float:
        """Return the elapsed wallclock time.

        Since the initialization of the TAMS object [seconds].

        Returns:
           TAMS elapse time.
        """
        return (datetime.datetime.utcnow() - self._startDate).total_seconds()

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


    def init_trajectory_pool(self) -> None:
        """Initialize the trajectory pool.

        Append the requested number of trajectories to the database.
        Trajectories are initialized but not advanced.
        """
        for n in range(self._tdb.nTraj()):
            T = Trajectory(
                    fmodel_t=self._fmodel_t,
                    parameters=self._parameters,
                    trajId=n,
                )
            T.setCheckFile(f"./{self._tdb.name()}/trajectories/{T.idstr()}.xml")
            self._tdb.appendTraj(T)

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
        inf_msg = f"Creating the initial pool of {self._tdb.nTraj()} trajectories"
        _logger.info(inf_msg)

        with get_runner_type(self._parameters)(self._parameters,
                                               pool_worker,
                                               self._parameters.get("runner",{}).get("nworker_init", 1)) as runner:
            for T in self._tdb.trajList():
                task = [T,
                        self._endDate,
                        self._tdb.save(),
                        self._tdb.name()]
                runner.make_promise(task)

            try:
                t_list = runner.execute_promises()
            except:
                err_msg = f"Failed to generate the initial pool of {self._tdb.nTraj()} trajectories"
                _logger.error(err_msg)
                raise

        # Re-order list since runner does not guarantee order
        # And update list of trajectories in the database
        t_list.sort(key=lambda t: t.id())
        self._tdb.updateTrajList(t_list)

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

    def check_exit_splitting_loop(self, k: int) -> tuple[bool, np.ndarray]:
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
        allConverged = True
        maxes = np.zeros(self._tdb.trajListLen())
        for i in range(self._tdb.trajListLen()):
            maxes[i] = self._tdb.getTraj(i).scoreMax()
            allConverged = allConverged and self._tdb.getTraj(i).isConverged()

        # Exit if our work is done
        if allConverged:
            inf_msg = f"All trajectories converged after {k} splitting iterations"
            _logger.info(inf_msg)
            return True, np.empty(1)

        # Exit if splitting is stalled
        if (np.amax(maxes) - np.amin(maxes)) < 1e-10:
            err_msg = f"Splitting is stalling with all trajectories stuck at a score_max: {np.amax(maxes)}"
            _logger.error(err_msg)
            raise TAMSError(err_msg)

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
            with get_runner_type(self._parameters)(self._parameters,
                                                   pool_worker,
                                                   self._parameters.get("runner",{}).get("nworker_iter", 1)) as runner:
                for i in ongoing_list:
                    T = self._tdb.getTraj(i)
                    task = [T, self._endDate, self._tdb.save(), self._tdb.name()]
                    runner.make_promise(task)
                finished_traj = runner.execute_promises()

                for T in finished_traj:
                    self._tdb.overwriteTraj(T.id(), T)

                # Clear list of ongoing branches
                self._tdb.reset_ongoing()

                # Increment splitting index
                k = self._tdb.kSplit() + runner.n_workers()
                self._tdb.setKSplit(k)
                self._tdb.saveSplittingData()


    def get_restart_at_random(self,
                              min_idx_list : list[int]) -> list[int]:
        """Get a list of trajectory index to restart from at random.

        Select trajectories to restart from among the ones not
        in min_idx_list.

        Args:
            min_idx_list: list of trajectory index to restart from

        Returns:
            list of trajectory index to restart from
        """
        # Enable deterministic runs by setting the a (different) seed
        # for each splitting iteration
        if self._parameters.get("tams",{}).get("deterministic", False):
            rng = np.random.default_rng(seed=42*self._tdb.kSplit())
        else:
            rng = np.random.default_rng()
        rest_idx = [-1] * len(min_idx_list)
        for i in range(len(min_idx_list)):
            rest_idx[i] = min_idx_list[0]
            while rest_idx[i] in min_idx_list:
                rest_idx[i] = rng.integers(0, self._tdb.trajListLen())
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
        k = self._tdb.kSplit()

        with get_runner_type(self._parameters)(self._parameters,
                                               ms_worker,
                                               self._parameters.get("runner",{}).get("nworker_iter", 1)) as runner:
            while k <= self._tdb.nSplitIter():
                inf_msg = f"Starting TAMS iter. {k} with {runner.n_workers()} workers"
                _logger.info(inf_msg)
                # Check for early exit conditions
                early_exit, maxes = self.check_exit_splitting_loop(k)
                if early_exit:
                    break

                # Plot trajectory database scores
                if self._plot_diags:
                    pltfile = f"Score_k{k:05}.png"
                    self._tdb.plotScoreFunctions(pltfile)

                # Get the nworker lower scored trajectories
                # or more if equal score
                min_idx_list, min_vals = get_min_scored(maxes, runner.n_workers())

                # Randomly select trajectory to branch from
                rest_idx = self.get_restart_at_random(min_idx_list)

                self._tdb.appendBias(len(min_idx_list))
                self._tdb.appendWeight(self._tdb.weights()[-1] * (1 - self._tdb.biases()[-1] / self._tdb.nTraj()))

                # Assemble a list of promises
                for i in range(len(min_idx_list)):
                    task = [self._tdb.getTraj(rest_idx[i]),
                            self._tdb.getTraj(min_idx_list[i]).id(),
                            min_vals[i],
                            self._endDate,
                            self._tdb.save(),
                            self._tdb.name()]
                    runner.make_promise(task)
                restartedTrajs = runner.execute_promises()

                # Update the trajectory database
                for T in restartedTrajs:
                    self._tdb.overwriteTraj(T.id(), T)

                if self.out_of_time():
                    # Save splitting data with ongoing trajectories
                    # but do not increment splitting index yet
                    self._tdb.saveSplittingData(min_idx_list)

                else:
                    # Update the trajectory database, increment splitting index
                    k = k + runner.n_workers()
                    self._tdb.setKSplit(k)
                    self._tdb.saveSplittingData()

    def compute_probability(self) -> float:
        """Compute the probability using TAMS.

        Returns:
            the transition probability
        """
        inf_msg = f"Computing {self._fmodel_t.name()} rare event probability using TAMS"
        _logger.info(inf_msg)

        # Skip pool stage if splitting iterative
        # process has started
        skip_pool = self._tdb.kSplit() > 0

        # Generate the initial trajectory pool
        if not skip_pool:
            self.generate_trajectory_pool()

        # Check for early convergence
        allConverged = True
        for T in self._tdb.trajList():
            if not T.isConverged():
                allConverged = False
                break

        if not skip_pool and allConverged:
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
        self.do_multilevel_splitting()

        if self.out_of_time():
            warn_msg = "Ran out of walltime ! Exiting now."
            _logger.warning(warn_msg)
            return -1.0

        W = self._tdb.nTraj() * self._tdb.weights()[-1]
        for i in range(len(self._tdb.biases())):
            W += self._tdb.biases()[i] * self._tdb.weights()[i]

        # Compute how many traj. converged
        successCount = 0
        for T in self._tdb.trajList():
            if T.isConverged():
                successCount += 1

        trans_prob = successCount * self._tdb.weights()[-1] / W

        inf_msg = f"Run time: {self.elapsed_time()} s"
        _logger.info(inf_msg)

        return trans_prob
