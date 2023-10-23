import copy
import time
import dask
import numpy as np
from dask.distributed import Client
from pytams.trajectory import Trajectory


class TAMS:
    """A class implementing TAMS.

    Hold a Trajectory database and mechanisms to
    populate the database.
    """

    def __init__(self, fmodel, parameters):
        """Initialize a TAMS run.

        Args:
            fmodel: the forward model
            parameters: a dictionary of input parameters
        """
        self.fmodel = fmodel
        self.parameters = parameters

        self.trajs_db = []
        self.hasEnded = None

        self.v = parameters.get("Verbose", False)

        self.nTraj = self.parameters.get("nTrajcetories", 500)
        self.nSplitIter = self.parameters.get("nSplitIter", 2000)

        self.nProc = self.parameters.get("nProc", 1)

        self._wallTime = self.parameters.get("wallTime", 600.0)
        self._startTime = time.monotonic()

    def elapsed_walltime(self) -> float:
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
           TAMS remaining time.
        """
        return self._wallTime - self.elapsed_walltime()

    def init_trajectory_pool(self):
        """Initialize the trajectory pool."""
        self.hasEnded = np.full((self.nTraj), False)
        for n in range(self.nTraj):
            self.trajs_db.append(
                Trajectory(
                    fmodel=self.fmodel,
                    parameters=self.parameters,
                    trajId="traj{:06}".format(n),
                )
            )

    def task_delayed(self, traj):
        """A worker to generate each initial trajectory.

        Args:
            traj: a trajectory
        """
        if self.remaining_walltime() > 0.05 * self._wallTime:
            traj.advance()
        return traj

    def generate_trajectory_pool(self):
        """Schedule the generation of a pool of stochastic trajectories."""
        if self.v:
            print("Creating the initial pool of {} trajectories".format(self.nTraj))

        self.init_trajectory_pool()

        with Client(threads_per_worker=1, n_workers=self.nProc):
            tasks_p = []
            for T in self.trajs_db:
                lazy_result = dask.delayed(self.task_delayed)(T)
                tasks_p.append(lazy_result)

            self.trajs_db = list(dask.compute(*tasks_p))

        if self.v:
            print("Run time: {} s".format(self.elapsed_walltime()))

    def worker(self, t_end, min_idx_list, min_val):
        """A worker to restart trajectories.

        Args:
            t_end: a final time
            min_idx_list: the list of trajectory restarted in
                          the current splitting iteration
            min_val: the value of the score function to restart from
        """
        rng = np.random.default_rng()
        rest_idx = min_idx_list[0]
        while rest_idx in min_idx_list:
            rest_idx = rng.integers(0,len(self.trajs_db))

        traj = Trajectory.restartFromTraj(self.trajs_db[rest_idx], min_val)

        traj.advance(t_end)

        return traj

    def do_multilevel_splitting(self):
        """Schedule splitting of the initial pool of stochastic trajectories."""
        if self.v:
            print("Using multi-level splitting to get the probability")

        l_bias = []
        weights = [1]

        # Check for early convergence
        allConverged = True
        for T in self.trajs_db:
            if not T.isConverged():
                allConverged = False
                break

        if allConverged:
            print("All trajectory converged prior to splitting !")
            return l_bias, weights

        with Client(threads_per_worker=1, n_workers=self.nProc):
            for _ in range(int(self.nSplitIter / self.nProc)):
                maxes = np.zeros(len(self.trajs_db))

                for i in range(len(self.trajs_db)):
                    maxes[i] = self.trajs_db[i].scoreMax()

                min_idx_list = np.argpartition(maxes, self.nProc)[: self.nProc]
                min_vals = maxes[min_idx_list]

                l_bias.append(len(min_idx_list))
                weights.append(weights[-1] * (1 - l_bias[-1] / self.nTraj))

                tasks_p = []
                for i in range(len(min_idx_list)):
                    lazy_result = dask.delayed(self.worker)(
                        19, min_idx_list, min_vals[i]
                    )
                    tasks_p.append(lazy_result)

                restartedTrajs = dask.compute(*tasks_p)

                for i in range(len(min_idx_list)):
                    self.trajs_db[min_idx_list[i]] = copy.deepcopy(restartedTrajs[i])

        return l_bias, weights

    def compute_probability(self) -> float:
        """Compute the probability using TAMS.

        Returns:
            the transition probability
        """
        if self.v:
            print("Computing rare event probability using TAMS")

        self.generate_trajectory_pool()

        l_bias, weights = self.do_multilevel_splitting()

        W = self.nTraj * weights[-1]
        for i in range(len(l_bias)):
            W += l_bias[i] * weights[i]

        # Compute how many traj. converged to the vicinity of B
        successCount = 0
        for T in self.trajs_db:
            if T.isConverged():
                successCount += 1

        trans_prob = successCount * weights[-1] / W

        if self.v:
            print("Run time: {} s".format(self.elapsed_walltime()))

        return trans_prob
