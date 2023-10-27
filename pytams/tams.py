import copy
import os
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List
from typing import Tuple
import dask
import numpy as np
from dask.distributed import Client
from pytams.trajectory import Trajectory
from pytams.xmlutils import dict_to_xml
from pytams.xmlutils import new_element


class TAMS:
    """A class implementing TAMS.

    Hold a Trajectory database and mechanisms to
    populate, explore and IO the database.
    """

    def __init__(self, fmodel, parameters: dict) -> None:
        """Initialize a TAMS run.

        Args:
            fmodel: the forward model
            parameters: a dictionary of input parameters
        """
        self._fmodel = fmodel
        self.parameters = parameters

        # Parse user-inputs
        self.v = parameters.get("Verbose", False)

        self._saveDB = self.parameters.get("DB_save", False)
        self._prefixDB = self.parameters.get("DB_prefix", "TAMS")
        self._restartDB = self.parameters.get("DB_restart", None)

        self._nTraj = self.parameters.get("nTrajectories", 500)
        self._nSplitIter = self.parameters.get("nSplitIter", 2000)

        self._nProc = self.parameters.get("nProc", 1)

        self._wallTime = self.parameters.get("wallTime", 600.0)

        # Data
        self._trajs_db = []
        self._hasEnded = None
        self._nameDB = "{}.tdb".format(self._prefixDB)

        # Initialize
        self._startTime = time.monotonic()
        if self._restartDB is not None:
            self.restoreTrajDB()
        else:
            self.initTrajDB()

    def initTrajDB(self) -> None:
        """Initialize the trajectory database."""
        if self._saveDB:
            if os.path.exists(self._nameDB):
                rng = np.random.default_rng()
                random_int = rng.integers(0, 999999)
                nameDB_rnd = "{}_{:06d}".format(self._nameDB, random_int)

                print(
                    """
                    TAMS database {} already present but not specified as restart.
                    It will be copied to {}.""".format(
                        self._nameDB, nameDB_rnd
                    )
                )
                shutil.move(self._nameDB, nameDB_rnd)

            os.mkdir(self._nameDB)

            # Header file with metadata
            headerFile = "{}/header.xml".format(self._nameDB)
            root = ET.Element("header")
            root.append(new_element("pyTAMS_version", datetime.now()))
            root.append(new_element("date", datetime.now()))
            root.append(new_element("model", self._fmodel.name()))
            root.append(dict_to_xml("parameters", self.parameters))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(headerFile)

            # Dynamically updated file with trajectory pool
            # Empty for now
            databaseFile = "{}/trajPool.xml".format(self._nameDB)
            root = ET.Element("trajectories")
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)

            # Empty trajectories subfolder
            os.mkdir("{}/{}".format(self._nameDB, "trajectories"))

    def appendTrajsToDB(self) -> None:
        """Append started trajectories to the pool file."""
        if self._saveDB:
            databaseFile = "{}/trajPool.xml".format(self._nameDB)
            tree = ET.parse(databaseFile)
            root = tree.getroot()
            for T in self._trajs_db:
                T_entry = root.find(T.id())
                if T.hasStarted() and T_entry is not None:
                    loc = T.checkFile()
                    root.append(new_element(T.id(), loc))

            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)

    def restoreTrajDB(self):
        """Initialize TAMS from a stored trajectory database."""
        pass

    def verbosePrint(self, message: str) -> None:
        """Print only in verbose mode."""
        if self.v:
            print("TAMS-[{}]".format(message))

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
        self.hasEnded = np.full((self._nTraj), False)
        for n in range(self._nTraj):
            self._trajs_db.append(
                Trajectory(
                    fmodel=self._fmodel,
                    parameters=self.parameters,
                    trajId="traj{:06}".format(n),
                )
            )

    def task_delayed(self, traj: Trajectory) -> Trajectory:
        """A worker to generate each initial trajectory.

        Args:
            traj: a trajectory
        """
        if self.remaining_walltime() > 0.05 * self._wallTime:
            traj.advance(walltime=self.remaining_walltime())
            if self._saveDB:
                traj.setCheckFile(
                    "{}/{}/{}.xml".format(self._nameDB, "trajectories", traj.id())
                )
                traj.store()

        return traj

    def generate_trajectory_pool(self) -> None:
        """Schedule the generation of a pool of stochastic trajectories."""
        self.verbosePrint(
            "Creating the initial pool of {} trajectories".format(self._nTraj)
        )

        self.init_trajectory_pool()

        with Client(threads_per_worker=1, n_workers=self._nProc):
            tasks_p = []
            for T in self._trajs_db:
                lazy_result = dask.delayed(self.task_delayed)(T)
                tasks_p.append(lazy_result)

            self._trajs_db = list(dask.compute(*tasks_p))

        # Update the trajectory database
        self.appendTrajsToDB()

        self.verbosePrint("Run time: {} s".format(self.elapsed_walltime()))

    def worker(
        self, t_end: float, min_idx_list: List[int], min_val: float
    ) -> Trajectory:
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
            rest_idx = rng.integers(0, len(self._trajs_db))

        traj = Trajectory.restartFromTraj(self._trajs_db[rest_idx], min_val)

        traj.advance(walltime=self.remaining_walltime())

        return traj

    def do_multilevel_splitting(self) -> Tuple[List[float], List[float]]:
        """Schedule splitting of the initial pool of stochastic trajectories."""
        self.verbosePrint("Using multi-level splitting to get the probability")

        l_bias = []
        weights = [1]

        # Check for early convergence
        allConverged = True
        for T in self._trajs_db:
            if not T.isConverged():
                allConverged = False
                break

        if allConverged:
            self.verbosePrint("All trajectory converged prior to splitting !")
            return l_bias, weights

        with Client(threads_per_worker=1, n_workers=self._nProc):
            for k in range(int(self._nSplitIter / self._nProc)):
                # Gather max score from all trajectories
                # and check for early convergence
                allConverged = True
                maxes = np.zeros(len(self._trajs_db))
                for i in range(len(self._trajs_db)):
                    maxes[i] = self._trajs_db[i].scoreMax()
                    allConverged = allConverged and self._trajs_db[i].isConverged()

                # Exit if our work is done
                if allConverged:
                    self.verbosePrint(
                        "All trajectory converged after {} splitting iterations".format(
                            k
                        )
                    )
                    break

                # Exit if splitting is stalled
                if (np.amax(maxes) - np.amin(maxes)) < 1e-10:
                    self.verbosePrint(
                        "Splitting is stalling with all trajectories stuck at a score_max: {}".format(
                            np.amax(maxes)
                        )
                    )
                    break

                # Get the nProc lower scored trajectories
                min_idx_list = np.argpartition(maxes, self._nProc)[: self._nProc]
                min_vals = maxes[min_idx_list]

                l_bias.append(len(min_idx_list))
                weights.append(weights[-1] * (1 - l_bias[-1] / self._nTraj))

                tasks_p = []
                for i in range(len(min_idx_list)):
                    lazy_result = dask.delayed(self.worker)(
                        19, min_idx_list, min_vals[i]
                    )
                    tasks_p.append(lazy_result)

                restartedTrajs = dask.compute(*tasks_p)

                for i in range(len(min_idx_list)):
                    self._trajs_db[min_idx_list[i]] = copy.deepcopy(restartedTrajs[i])

        return l_bias, weights

    def compute_probability(self) -> float:
        """Compute the probability using TAMS.

        Returns:
            the transition probability
        """
        self.verbosePrint("Computing rare event probability using TAMS")

        self.generate_trajectory_pool()

        l_bias, weights = self.do_multilevel_splitting()

        W = self._nTraj * weights[-1]
        for i in range(len(l_bias)):
            W += l_bias[i] * weights[i]

        # Compute how many traj. converged to the vicinity of B
        successCount = 0
        for T in self._trajs_db:
            if T.isConverged():
                successCount += 1

        trans_prob = successCount * weights[-1] / W

        self.verbosePrint("Run time: {} s".format(self.elapsed_walltime()))

        return trans_prob

    def nTraj(self) -> int:
        """Return the number of trajectory used for TAMS."""
        return self._nTraj
