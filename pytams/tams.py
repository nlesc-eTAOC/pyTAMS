import copy
import os
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List
from typing import Tuple
import numpy as np
from pytams.daskutils import DaskRunner
from pytams.trajectory import Trajectory
from pytams.xmlutils import dict_to_xml
from pytams.xmlutils import new_element
from pytams.xmlutils import xml_to_dict


class TAMSError(Exception):
    """Exception class for TAMS."""

    pass


class TAMS:
    """A class implementing TAMS.

    Hold a Trajectory database and mechanisms to
    populate, explore and IO the database.
    """

    def __init__(self, fmodel_t, parameters: dict) -> None:
        """Initialize a TAMS run.

        Args:
            fmodel_t: the forward model type
            parameters: a dictionary of input parameters
        """
        self._fmodel_t = fmodel_t
        self.parameters = parameters

        # Parse user-inputs
        self.v = parameters.get("Verbose", False)

        self._saveDB = self.parameters.get("DB_save", False)
        self._prefixDB = self.parameters.get("DB_prefix", "TAMS")
        self._restartDB = self.parameters.get("DB_restart", None)

        self._nTraj = self.parameters.get("nTrajectories", 500)
        self._nSplitIter = self.parameters.get("nSplitIter", 2000)
        self._wallTime = self.parameters.get("wallTime", 600.0)

        # Data
        self._trajs_db = []
        self._hasEnded = None
        self._nameDB = "{}.tdb".format(self._prefixDB)

        # Splitting data
        self._kSplit = 0
        self._l_bias = []
        self._weights = [1]

        # Initialize
        self._startTime = time.monotonic()
        if self._restartDB is not None:
            self.restoreTrajDB()
        else:
            self.initTrajDB()
            self.init_trajectory_pool()

    def initTrajDB(self) -> None:
        """Initialize the trajectory database."""
        if self._saveDB:
            self.verbosePrint(
                "Initializing the trajectories database {}".format(self._nameDB)
            )
            if os.path.exists(self._nameDB) and self._nameDB != self._restartDB:
                rng = np.random.default_rng(12345)
                copy_exists = True
                while copy_exists:
                    random_int = rng.integers(0, 999999)
                    nameDB_rnd = "{}_{:06d}".format(self._nameDB, random_int)
                    copy_exists = os.path.exists(nameDB_rnd)

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
            mdata = ET.SubElement(root, "metadata")
            mdata.append(new_element("pyTAMS_version", datetime.now()))
            mdata.append(new_element("date", datetime.now()))
            mdata.append(new_element("model_t", self._fmodel_t.name()))
            root.append(dict_to_xml("parameters", self.parameters))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(headerFile)

            # Initialialize splitting data file
            self.saveSplittingData(self._nameDB)

            # Dynamically updated file with trajectory pool
            # Empty for now
            databaseFile = "{}/trajPool.xml".format(self._nameDB)
            root = ET.Element("trajectories")
            root.append(new_element("nTraj", self._nTraj))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)

            # Empty trajectories subfolder
            os.mkdir("{}/{}".format(self._nameDB, "trajectories"))

    def appendTrajsToDB(self) -> None:
        """Append started trajectories to the pool file."""
        if self._saveDB:
            self.verbosePrint(
                "Appending started trajectories to database {}".format(self._nameDB)
            )
            databaseFile = "{}/trajPool.xml".format(self._nameDB)
            tree = ET.parse(databaseFile)
            root = tree.getroot()
            for T in self._trajs_db:
                T_entry = root.find(T.id())
                if T.hasStarted() and T_entry is None:
                    loc = T.checkFile()
                    root.append(new_element(T.id(), loc))

            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)

    def saveSplittingData(self, a_db: str) -> None:
        """Write splitting data to XML file."""
        # Splitting data file
        splittingDataFile = "{}/splittingData.xml".format(a_db)
        root = ET.Element("Splitting")
        root.append(new_element("kSplit", self._kSplit))
        root.append(new_element("bias", np.array(self._l_bias)))
        root.append(new_element("weight", np.array(self._weights)))
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(splittingDataFile)

    def readSplittingData(self, a_db: str) -> None:
        """Read splitting data from XML file."""
        # Read data file
        splittingDataFile = "{}/splittingData.xml".format(a_db)
        tree = ET.parse(splittingDataFile)
        root = tree.getroot()
        datafromxml = xml_to_dict(root)
        self._kSplit = datafromxml["kSplit"]
        self._l_bias = datafromxml["bias"].tolist()
        self._weights = datafromxml["weight"].tolist()

    def restoreTrajDB(self) -> None:
        """Initialize TAMS from a stored trajectory database."""
        if os.path.exists(self._restartDB):
            self.verbosePrint(
                "Restoring from the trajectories database {}".format(self._restartDB)
            )

            # Check the database parameters against current run
            self.check_database_consistency()

            # Load splitting data
            self.readSplittingData(self._restartDB)

            # Init trajectory pool and load trajectory stored
            # in the database when available.
            dbFile = "{}/trajPool.xml".format(self._restartDB)
            tree = ET.parse(dbFile)
            root = tree.getroot()
            for n in range(self._nTraj):
                trajId = "traj{:06}".format(n)
                T_entry = root.find(trajId)
                if T_entry is not None:
                    chkFile = T_entry.text
                    if os.path.exists(chkFile):
                        self._trajs_db.append(
                            Trajectory.restoreFromChk(
                                chkFile,
                                fmodel_t=self._fmodel_t,
                            )
                        )
                    else:
                        raise TAMSError(
                            "Could not find the trajectory checkFile {} listed in the TAMS database !".format(
                                chkFile
                            )
                        )
                else:
                    self._trajs_db.append(
                        Trajectory(
                            fmodel_t=self._fmodel_t,
                            parameters=self.parameters,
                            trajId="traj{:06}".format(n),
                        )
                    )

        else:
            raise TAMSError(
                "Could not find the {} TAMS database !".format(self._restartDB)
            )

    def check_database_consistency(self) -> None:
        """Check the restart database consistency."""
        # Open and load header
        headerFile = "{}/header.xml".format(self._restartDB)
        tree = ET.parse(headerFile)
        root = tree.getroot()
        headerfromxml = xml_to_dict(root.find("metadata"))
        if self._fmodel_t.name() != headerfromxml["model_t"]:
            raise TAMSError(
                "Trying to restore a TAMS with {} model from database with {} model !".format(
                    self._fmodel_t.name(), headerfromxml["model_t"]
                )
            )

        # Parameters stored in the database override any
        # newly modified params
        # TODO: will need to relax this later on
        paramsfromxml = xml_to_dict(root.find("parameters"))
        self.parameters.update(paramsfromxml)

    def verbosePrint(self, message: str) -> None:
        """Print only in verbose mode."""
        if self.v:
            print("TAMS-[{}]".format(message))

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
            self._trajs_db.append(
                Trajectory(
                    fmodel_t=self._fmodel_t,
                    parameters=self.parameters,
                    trajId="traj{:06}".format(n),
                )
            )

    def task_delayed(self, traj: Trajectory) -> Trajectory:
        """A worker to generate each initial trajectory.

        Args:
            traj: a trajectory
        """
        if not self.out_of_time():
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

        with DaskRunner(self.parameters) as runner:
            tasks_p = []
            for T in self._trajs_db:
                if not T.hasEnded():
                    tasks_p.append(runner.make_promise(self.task_delayed, T))

            self._trajs_db = runner.execute_promises(tasks_p)

        # Update the trajectory database
        self.appendTrajsToDB()

        self.verbosePrint("Run time: {} s".format(self.elapsed_time()))

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
        rng = np.random.default_rng(12345)
        rest_idx = min_idx_list[0]
        while rest_idx in min_idx_list:
            rest_idx = rng.integers(0, len(self._trajs_db))

        traj = Trajectory.restartFromTraj(self._trajs_db[rest_idx], min_val)

        traj.advance(walltime=self.remaining_walltime())

        return traj

    def do_multilevel_splitting(self) -> None:
        """Schedule splitting of the initial pool of stochastic trajectories."""
        self.verbosePrint("Using multi-level splitting to get the probability")

        # Initialize splitting iterations counter
        k = self._kSplit

        with DaskRunner(self.parameters) as runner:
            while k <= self._nSplitIter:
                # Check for walltime
                if self.out_of_time():
                    self.verbosePrint(
                        "Ran out of time after {} splitting iterations".format(
                            k
                        )
                    )
                    break

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

                # Get the nworker lower scored trajectories
                min_idx_list = np.argpartition(maxes, runner.dask_nworker)[
                    : runner.dask_nworker
                ]
                min_vals = maxes[min_idx_list]

                self._l_bias.append(len(min_idx_list))
                self._weights.append(self._weights[-1] * (1 - self._l_bias[-1] / self._nTraj))

                # Assemble a list of promises
                tasks_p = []
                for i in range(len(min_idx_list)):
                    tasks_p.append(
                        runner.make_promise(self.worker, 19, min_idx_list, min_vals[i])
                    )

                restartedTrajs = runner.execute_promises(tasks_p)

                # Update the trajectory pool and database
                k += runner.dask_nworker
                self._kSplit = k
                for i in range(len(min_idx_list)):
                    self._trajs_db[min_idx_list[i]] = copy.deepcopy(restartedTrajs[i])

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
        skip_pool = self._kSplit > 0

        # Generate the initial trajectory pool
        if not skip_pool:
            self.generate_trajectory_pool()

        # Check for early convergence
        allConverged = True
        for T in self._trajs_db:
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

        W = self._nTraj * self._weights[-1]
        for i in range(len(self._l_bias)):
            W += self._l_bias[i] * self._weights[i]

        # Compute how many traj. converged to the vicinity of B
        successCount = 0
        for T in self._trajs_db:
            if T.isConverged():
                successCount += 1

        trans_prob = successCount * self._weights[-1] / W

        self.verbosePrint("Run time: {} s".format(self.elapsed_time()))

        return trans_prob

    def nTraj(self) -> int:
        """Return the number of trajectory used for TAMS."""
        return self._nTraj
