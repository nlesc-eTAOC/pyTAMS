import argparse
import asyncio
import concurrent.futures
import functools
import os
import time
from typing import Any
from typing import Optional
from typing import Tuple
import numpy as np
import toml
from pytams.database import Database
from pytams.taskrunner import get_runner_type
from pytams.trajectory import Trajectory
from pytams.trajectory import WallTimeLimit


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

    Hold a Trajectory database and mechanisms to
    populate, explore and IO the database.
    """

    def __init__(self,
                 fmodel_t : Any,
                 a_args: Optional[list[str]] = None) -> None:
        """Initialize a TAMS run.

        Args:
            fmodel_t: the forward model type
            a_args: optional list of options
        """
        self._fmodel_t = fmodel_t

        input_file = vars(parse_cl_args(a_args=a_args))["input"]
        if (not os.path.exists(input_file)):
            raise TAMSError(
                "Could not find the {} TAMS input file !".format(input_file)
            )

        with open(input_file, 'r') as f:
            self.parameters = toml.load(f)

        # Parse user-inputs
        self.v = self.parameters["tams"].get("verbose", False)
        self._nTraj : int = self.parameters["tams"].get("ntrajectories", 500)
        self._nSplitIter : int = self.parameters["tams"].get("nsplititer", 2000)
        self._wallTime : float = self.parameters["tams"].get("walltime", 24.0*3600.0)
        self._plot_diags = self.parameters["tams"].get("diagnostics", False)

        # Database
        self._tdb = Database(fmodel_t,
                             self.parameters,
                             self._nTraj,
                             self._nSplitIter)

        # Initialize
        self._startTime : float = time.monotonic()
        if self._tdb.isEmpty():
            self.init_trajectory_pool()

    def nTraj(self) -> int:
        """Return the number of trajectory used for TAMS."""
        return self._nTraj

    def verbosePrint(self, message: str) -> None:
        """Print only in verbose mode."""
        if self.v:
            print("TAMS-[{}]".format(message), flush=True)

    def elapsed_time(self) -> float:
        """Return the elapsed wallclock time.

        Since the initialization of TAMS [seconds].

        Returns:
           TAMS elapse time.
        """
        return time.monotonic() - self._startTime

    def remaining_walltime(self) -> float:
        """Return the remaining wallclock time.

        [seconds]

        Returns:
           TAMS remaining wall time.
        """
        return self._wallTime - self.elapsed_time()

    def out_of_time(self) -> bool:
        """Return true if insufficient walltime remains.

        Returns:
           boolean indicating wall time availability.
        """
        return self.remaining_walltime() < 0.05 * self._wallTime


    def init_trajectory_pool(self) -> None:
        """Initialize the trajectory pool."""
        self.hasEnded = np.full((self._nTraj), False)
        for n in range(self._nTraj):
            self._tdb.appendTraj(
                Trajectory(
                    fmodel_t=self._fmodel_t,
                    parameters=self.parameters,
                    trajId=n,
                )
            )

    def generate_trajectory_pool(self) -> None:
        """Schedule the generation of a pool of stochastic trajectories."""
        self.verbosePrint(
            "Creating the initial pool of {} trajectories".format(self._nTraj)
        )

        t_list = []
        with get_runner_type(self.parameters)(self.parameters,
                                              pool_worker,
                                              pool_worker_async,
                                              self.parameters.get("runner",{}).get("nworker_init", 1)) as runner:
            for T in self._tdb.trajList():
                task = [T, self._startTime+self._wallTime, self._tdb.save(), self._tdb.name()]
                runner.make_promise(task)
            t_list = runner.execute_promises()

        # Re-order list since runner does not guarantee order
        t_list.sort(key=lambda t: t.id())
        self._tdb.updateTrajList(t_list)
        self._tdb.updateDiskData()

        self.verbosePrint("Run time: {} s".format(self.elapsed_time()))

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
            self.verbosePrint(
                "Ran out of time after {} splitting iterations".format(
                    k
                )
            )
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
            self.verbosePrint(
                "All trajectory converged after {} splitting iterations".format(
                    k
                )
            )
            return True, np.empty(1)

        # Exit if splitting is stalled
        if (np.amax(maxes) - np.amin(maxes)) < 1e-10:
            raise TAMSError(
                "Splitting is stalling with all trajectories stuck at a score_max: {}".format(
                    np.amax(maxes))
            )

        return False, maxes

    def finished_ongoing_splitting(self) -> None:
        """Check and finish unfinished splitting iterations."""
        # Check the database for unfinished splitting iteration when restarting.
        # At this point, branching has been done, but advancing to final
        # time is still ongoing.
        ongoing_list = self._tdb.get_ongoing()
        if ongoing_list:
            print("Unfinished splitting iteration detected, traj {} need(s) finishing".format(self._tdb.get_ongoing()))
            with get_runner_type(self.parameters)(self.parameters,
                                                  pool_worker,
                                                  pool_worker_async,
                                                  self.parameters.get("runner",{}).get("nworker_init", 1)) as runner:
                for i in ongoing_list:
                    T = self._tdb.getTraj(i)
                    task = [T, self._startTime+self._wallTime, self._tdb.save(), self._tdb.name()]
                    runner.make_promise(task)
                finished_traj = runner.execute_promises()
                for T in finished_traj:
                    self._tdb.overwriteTraj(T.id(), T)

                # Clear list of ongoing branches
                self._tdb.reset_ongoing()

                # increment splitting index
                k = self._tdb.kSplit() + runner.n_workers()
                self._tdb.setKSplit(k)


    def get_restart_at_random(self, min_idx_list : list[int]) -> list[int]:
        """Get a list of trajectory index to restart from at random."""
        # Enable deterministic runs by setting the a (different) seed
        # for each splitting iteration
        if self.parameters.get("tams",{}).get("deterministic", False):
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
        """Schedule splitting of the initial pool of stochastic trajectories."""
        self.verbosePrint("Using multi-level splitting to get the probability")

        # Finish any unfinished splitting iteration
        self.finished_ongoing_splitting()

        # Initialize splitting iterations counter
        k = self._tdb.kSplit()

        with get_runner_type(self.parameters)(self.parameters,
                                              ms_worker,
                                              ms_worker_async,
                                              self.parameters.get("runner",{}).get("nworker_iter", 1)) as runner:
            while k <= self._nSplitIter:
                print("Starting it {} with {} workers".format(k, runner.n_workers()))
                # Check for early exit conditions
                early_exit, maxes = self.check_exit_splitting_loop(k)
                if early_exit:
                    break

                # Plot trajectory database scores
                if self._plot_diags:
                    pltfile = "Score_k{:05}.png".format(k)
                    self._tdb.plotScoreFunctions(pltfile)

                # Get the nworker lower scored trajectories
                min_idx_list = np.argpartition(maxes, runner.n_workers())[
                    : runner.n_workers()
                ]
                min_vals = maxes[min_idx_list]

                # Randomly select trajectory to branch from
                rest_idx = self.get_restart_at_random(min_idx_list.tolist())

                self._tdb.appendBias(len(min_idx_list))
                self._tdb.appendWeight(self._tdb.weights()[-1] * (1 - self._tdb.biases()[-1] / self._nTraj))

                # Assemble a list of promises
                for i in range(len(min_idx_list)):
                    task = [1.0e9,
                            self._tdb.getTraj(rest_idx[i]),
                            self._tdb.getTraj(min_idx_list[i]).id(),
                            min_vals[i],
                            self._startTime+self._wallTime,
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
                    self._tdb.saveSplittingData(min_idx_list.tolist())

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
        self.verbosePrint(
            "Computing {} rare event probability using TAMS".format(
                self._fmodel_t.name()
            )
        )

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
            self.verbosePrint("All trajectory converged prior to splitting !")
            return 1.0

        if self.out_of_time():
            self.verbosePrint("Ran out of walltime ! Exiting now.")
            return -1.0

        # Perform multilevel splitting
        self.do_multilevel_splitting()

        if self.out_of_time():
            self.verbosePrint("Ran out of walltime ! Exiting now.")
            return -1.0

        W = self._nTraj * self._tdb.weights()[-1]
        for i in range(len(self._tdb.biases())):
            W += self._tdb.biases()[i] * self._tdb.weights()[i]

        # Compute how many traj. converged
        successCount = 0
        for T in self._tdb.trajList():
            if T.isConverged():
                successCount += 1

        trans_prob = successCount * self._tdb.weights()[-1] / W

        self.verbosePrint("Run time: {} s".format(self.elapsed_time()))

        return trans_prob

def pool_worker(traj: Trajectory,
                 wall_time_info: float,
                 saveDB: bool,
                 nameDB: str) -> Trajectory:
    """A worker to generate each initial trajectory.

    Args:
        traj: a trajectory
        wall_time_info: the time limit to advance the trajectory
        saveDB: a bool to save the trajectory to database
        nameDB: name of the database

    Returns:
        The updated trajectory
    """
    wall_time = wall_time_info - time.monotonic()
    if wall_time > 0.0 and not traj.hasEnded():
        print("Advancing {} [time left: {}]".format(traj.idstr(), wall_time), flush=True)
        if saveDB:
            traj.setCheckFile(
                "{}/{}/{}.xml".format(nameDB, "trajectories", traj.idstr())
            )
        try:
            traj.advance(walltime=wall_time)
        except WallTimeLimit:
            print("Trajectory advance ran out of time !")
            if saveDB:
                traj.store()
            raise
        except Exception:
            print("Advance ran into an error !")
            raise
        else:
            if saveDB:
                traj.store()

    return traj


async def pool_worker_async(
    queue : asyncio.Queue[Tuple[Trajectory, float, bool, str]],
    res_queue : asyncio.Queue[Trajectory],
    executor : concurrent.futures.Executor) -> None:
    """A worker to generate each initial trajectory.

    Args:
        queue: a queue from which to get tasks
        res_queue: a queue to put the results in
        executor: an executor to launch the work in
    """
    while True:
        traj, wall_time_info, saveDB, nameDB = await queue.get()
        loop = asyncio.get_running_loop()
        updated_traj = await loop.run_in_executor(
            executor,
            functools.partial(pool_worker, traj, wall_time_info, saveDB, nameDB)
        )
        await res_queue.put(updated_traj)
        queue.task_done()

def ms_worker(
    t_end: float,
    fromTraj: Trajectory,
    rstId: int,
    min_val: float,
    wall_time_info: float,
    saveDB: bool,
    nameDB: str,
) -> Trajectory:
    """A worker to restart trajectories.

    Args:
        t_end: a final time
        fromTraj: a trajectory to restart from
        rstId: Id of the trajectory being worked on
        min_val: the value of the score function to restart from
        wall_time_info: the time limit to advance the trajectory
        saveDB: a bool to save the trajectory to database
        nameDB: name of DB to save the traj in (Opt)
    """
    print("Restarting [{}] from {}".format(rstId, fromTraj.idstr(), ), flush=True)
    wall_time = wall_time_info - time.monotonic()
    traj = Trajectory.restartFromTraj(fromTraj, rstId, min_val)
    if saveDB:
        traj.setCheckFile(
            "{}/{}/{}.xml".format(nameDB, "trajectories", traj.idstr())
        )

    try:
        traj.advance(walltime=wall_time)
    except WallTimeLimit:
        print("Trajectory advance ran out of time !")
        if saveDB:
            traj.store()
        raise
    except Exception:
        print("Advance ran into an error !")
        raise
    else:
        if saveDB:
            traj.store()

    return traj

async def ms_worker_async(
    queue : asyncio.Queue[Any],
    res_queue : asyncio.Queue[Trajectory],
    executor : concurrent.futures.Executor,
) -> None:
    """An async worker to restart trajectories.

    Args:
        queue: an asyncio queue to get work from
        res_queue: an asyncio queue to put results in
        executor: an executor
    """
    while True:
        t_end, fromTraj, rstId, min_val, wall_time_info, saveDB, nameDB = await queue.get()
        loop = asyncio.get_running_loop()
        restarted_traj = await loop.run_in_executor(
            executor,
            functools.partial(ms_worker, t_end, fromTraj, rstId, min_val,
                              wall_time_info, saveDB, nameDB)
        )
        await res_queue.put(restarted_traj)
        queue.task_done()
