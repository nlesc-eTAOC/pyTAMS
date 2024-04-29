import argparse
import os
import time
import numpy as np
import toml
from pytams.daskutils import DaskRunner
from pytams.database import Database
from pytams.database import formTrajID
from pytams.trajectory import Trajectory


class TAMSError(Exception):
    """Exception class for TAMS."""

    pass

def parse_cl_args(a_args: list = None) -> argparse.Namespace :
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
                 fmodel_t,
                 a_args: list = None) -> None:
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
        self._nTraj = self.parameters["tams"].get("ntrajectories", 500)
        self._nSplitIter = self.parameters["tams"].get("nsplititer", 2000)
        self._wallTime = self.parameters["tams"].get("walltime", 24.0*3600.0)
        self._plot_diags = self.parameters["tams"].get("diagnostics", False)

        # Database
        self._tdb = Database(fmodel_t,
                             self.parameters,
                             self._nTraj,
                             self._nSplitIter)

        # Initialize
        self._startTime = time.monotonic()
        if self._tdb.isEmpty():
            self.init_trajectory_pool()


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


    def init_trajectory_pool(self):
        """Initialize the trajectory pool."""
        self.hasEnded = np.full((self._nTraj), False)
        for n in range(self._nTraj):
            self._tdb.appendTraj(
                Trajectory(
                    fmodel_t=self._fmodel_t,
                    parameters=self.parameters,
                    trajId=formTrajID(n),
                )
            )

    def generate_trajectory_pool(self) -> None:
        """Schedule the generation of a pool of stochastic trajectories."""
        self.verbosePrint(
            "Creating the initial pool of {} trajectories".format(self._nTraj)
        )

        with DaskRunner(self.parameters,
                        self.parameters.get("dask",{}).get("nworker_init", 1)) as runner:
            # Assemble a list of promises
            # All the trajectories are added, even those already done
            tasks_p = []
            for T in self._tdb.trajList():
                tasks_p.append(runner.make_promise(task_delayed,
                                                   T,
                                                   self.remaining_walltime(),
                                                   self._tdb._save,
                                                   self._tdb._name))

            self._tdb.updateTrajList(runner.execute_promises(tasks_p))

        # Update the trajectory stored database
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


    def do_multilevel_splitting(self) -> None:
        """Schedule splitting of the initial pool of stochastic trajectories."""
        self.verbosePrint("Using multi-level splitting to get the probability")

        # Initialize splitting iterations counter
        k = self._tdb.kSplit()

        with DaskRunner(self.parameters,
                        self.parameters.get("dask",{}).get("nworker_iter", 1)) as runner:
            while k <= self._nSplitIter:
                # Check for early exit conditions
                early_exit, maxes = self.check_exit_splitting_loop(k)
                if early_exit:
                    break

                # Plot trajectory database scores
                if self._plot_diags:
                    pltfile = "Score_k{:05}.png".format(k)
                    self._tdb.plotScoreFunctions(pltfile)

                # Get the nworker lower scored trajectories
                min_idx_list = np.argpartition(maxes, runner.dask_nworker)[
                    : runner.dask_nworker
                ]
                min_vals = maxes[min_idx_list]

                # Randomly select trajectory to branch from
                rng = np.random.default_rng()
                rest_idx = [-1] * len(min_idx_list)
                for i in range(len(min_idx_list)):
                    rest_idx[i] = min_idx_list[0]
                    while rest_idx[i] in min_idx_list:
                        rest_idx[i] = rng.integers(0, self._tdb.trajListLen())

                self._tdb.appendBias(len(min_idx_list))
                self._tdb.appendWeight(self._tdb.weights()[-1] * (1 - self._tdb.biases()[-1] / self._nTraj))

                # Assemble a list of promises
                tasks_p = []
                for i in range(len(min_idx_list)):
                    tasks_p.append(
                        runner.make_promise(worker,
                                            1.0e9,
                                            self._tdb.getTraj(rest_idx[i]),
                                            self._tdb.getTraj(min_idx_list[i]).id(),
                                            min_vals[i],
                                            self.remaining_walltime())
                    )
                restartedTrajs = runner.execute_promises(tasks_p)

                # Update the trajectory pool and database
                k = k + runner.dask_nworker
                self._tdb.setKSplit(k)
                for i in range(len(min_idx_list)):
                    self._tdb.overwriteTraj(min_idx_list[i],restartedTrajs[i])

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

    def nTraj(self) -> int:
        """Return the number of trajectory used for TAMS."""
        return self._nTraj


def task_delayed(traj: Trajectory,
                 wall_time: float,
                 saveDB: bool,
                 nameDB: str) -> Trajectory:
    """A worker to generate each initial trajectory.

    Args:
        traj: a trajectory
        wall_time: a time limit to advance the trajectory
        saveDB: a bool to save the trajectory to database
        nameDB: name of the database
    """
    if wall_time > 0.0 and not traj.hasEnded():
        print("Advancing {}".format(traj.id()), flush=True)
        traj.advance(walltime=wall_time)
        if saveDB:
            traj.setCheckFile(
                "{}/{}/{}.xml".format(nameDB, "trajectories", traj.id())
            )
            traj.store()

    return traj


def worker(
    t_end: float,
    fromTraj: Trajectory,
    rstId: str,
    min_val: float,
    wall_time: float,
) -> Trajectory:
    """A worker to restart trajectories.

    Args:
        t_end: a final time
        fromTraj: a trajectory to restart from
        rstId: Id of the trajectory being worked on
        min_val: the value of the score function to restart from
        wall_time: a time limit to advance the trajectory
    """
    traj = Trajectory.restartFromTraj(fromTraj, rstId, min_val)

    traj.advance(walltime=wall_time)

    return traj
