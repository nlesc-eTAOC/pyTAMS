import copy
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import toml
from pytams.trajectory import Trajectory
from pytams.xmlutils import new_element
from pytams.xmlutils import xml_to_dict


class DatabaseError(Exception):
    """Exception class for TAMS Database."""

    pass


def formTrajID(n: int) -> str:
    """Helper to assemble a trajectory ID string."""
    return "traj{:06}".format(n)


class Database:
    """A database class for TAMS."""

    def __init__(self,
                 fmodel_t,
                 params: dict,
                 ntraj: int = None,
                 nsplititer: int = None) -> None:
        """Init a TAMS database.

        Args:
            fmodel_t: the forward model type
            params: a dictionary of parameters
            ntraj: [OPT] number of traj to hold
            nsplititer: [OPT] number of splitting iteration to hold
        """
        self._fmodel_t = fmodel_t

        # Metadata
        self._name = None
        self._verbose = False
        self._save = False
        self._load = None
        self._format = "XML"
        self._parameters = params

        # Trajectory Pool
        self._trajs_db = []

        # Splitting data
        self._ksplit = 0
        self._l_bias = []
        self._weights = [1.0]

        self._save = params.get("database", {}).get("DB_save", False)
        self._prefix = params.get("database", {}).get("DB_prefix", "TAMS")
        self._load = params.get("database", {}).get("DB_restart", None)
        self._name = "{}.tdb".format(self._prefix)

        if self._load:
            self._name = self._load
            self._restoreTrajDB()
        else:
            if not ntraj:
                raise DatabaseError(
                        "Initializing TAMS database from scratch require ntraj !"
                )
            if not nsplititer:
                raise DatabaseError(
                        "Initializing TAMS database from scratch require nsplititer !"
                )
            self._ntraj = ntraj
            self._nsplititer = nsplititer
            self._setUpTree()

    def _setUpTree(self) -> None:
        """Initialize the trajectory database tree."""
        if self._save:
            if os.path.exists(self._name) and self._name != self._load:
                rng = np.random.default_rng(12345)
                copy_exists = True
                while copy_exists:
                    random_int = rng.integers(0, 999999)
                    name_rnd = "{}_{:06d}".format(self._name, random_int)
                    copy_exists = os.path.exists(name_rnd)
                print(
                    """
                    TAMS database {} already present but not specified as restart.
                    It will be copied to {}.""".format(
                        self._name, name_rnd
                    )
                )
                shutil.move(self._name, name_rnd)

            os.mkdir(self._name)

            # TODO: remove this, mixed format is weird
            with open("{}/input_params.toml".format(self._name), 'w') as f:
                toml.dump(self._parameters, f)

            # Header file with metadata
            self._writeMetadata()

            # Empty trajectories subfolder
            os.mkdir("{}/{}".format(self._name, "trajectories"))

    def _writeMetadata(self) -> None:
        """Write the database Metadata to disk."""
        if self._format == "XML":
            headerFile = "{}/header.xml".format(self._name)
            root = ET.Element("header")
            mdata = ET.SubElement(root, "metadata")
            mdata.append(new_element("pyTAMS_version", datetime.now()))
            mdata.append(new_element("date", datetime.now()))
            mdata.append(new_element("model_t", self._fmodel_t.name()))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(headerFile)

            # Initialialize splitting data file
            self.saveSplittingData()

            # Dynamically updated file with trajectory pool
            # Empty for now
            databaseFile = "{}/trajPool.xml".format(self._name)
            root = ET.Element("trajectories")
            root.append(new_element("ntraj", self._ntraj))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)
        else:
            raise DatabaseError(
                    "Unsupported TAMS database format: {} !".format(self._format)
            )

    def _readHeader(self) -> tuple[str, datetime, str]:
        if self._load:
            headerFile = "{}/header.xml".format(self._name)
            tree = ET.parse(headerFile)
            root = tree.getroot()
            mdata = root.find("metadata")
            datafromxml = xml_to_dict(mdata)
            pyTAMS_version = datafromxml["pyTAMS_version"]
            db_date = datafromxml["date"]
            db_model = datafromxml["model_t"]
            return pyTAMS_version, db_date, db_model
        else:
            return None, None, None


    def appendTrajsToDB(self) -> None:
        """Append started trajectories to the pool file."""
        if self._save:
            print(
                "Appending started trajectories to database {}".format(self._name)
            )
            databaseFile = "{}/trajPool.xml".format(self._name)
            tree = ET.parse(databaseFile)
            root = tree.getroot()
            for T in self._trajs_db:
                T_entry = root.find(T.id())
                if T.hasStarted() and T_entry is None:
                    loc = T.checkFile()
                    root.append(new_element(T.id(), loc))

            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)

    def saveSplittingData(self) -> None:
        """Write splitting data."""
        if not self._save:
            return

        # Splitting data file
        if self._format == "XML":
            splittingDataFile = "{}/splittingData.xml".format(self._name)
            root = ET.Element("splitting")
            root.append(new_element("nsplititer", self._nsplititer))
            root.append(new_element("ksplit", self._ksplit))
            root.append(new_element("bias", np.array(self._l_bias)))
            root.append(new_element("weight", np.array(self._weights)))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(splittingDataFile)
        else:
            raise DatabaseError(
                    "Unsupported TAMS database format: {} !".format(self._format)
            )

    def _readSplittingData(self) -> None:
        """Read splitting data."""
        # Read data file
        if self._format == "XML":
            splittingDataFile = "{}/splittingData.xml".format(self._name)
            tree = ET.parse(splittingDataFile)
            root = tree.getroot()
            datafromxml = xml_to_dict(root)
            self._nsplititer = datafromxml["nsplititer"]
            self._ksplit = datafromxml["ksplit"]
            self._l_bias = datafromxml["bias"].tolist()
            self._weights = datafromxml["weight"].tolist()
        else:
            raise DatabaseError(
                    "Unsupported TAMS database format: {} !".format(self._format)
            )

    def _restoreTrajDB(self) -> None:
        """Initialize TAMS from a stored trajectory database."""
        if os.path.exists(self._load):
            print("Load TAMS database: {}".format(self._load))

            # Check the database parameters against current run
            self._check_database_consistency()

            # Load splitting data
            self._readSplittingData()

            # Load trajectories stored in the database when available.
            dbFile = "{}/trajPool.xml".format(self._load)
            nTrajRestored = self.loadTrajectoryDB(dbFile)

            print("{} trajectories loaded".format(nTrajRestored))
        else:
            raise DatabaseError(
                "Could not find the {} TAMS database !".format(self._load)
            )

    def loadTrajectoryDB(self, dbFile: str) -> int:
        """Load trajectories stored into the database.

        Args:
            dbFile: the database file

        Return:
            number of trajectories loaded
        """
        # Counter for number of trajectory loaded
        nTrajRestored = 0

        tree = ET.parse(dbFile)
        root = tree.getroot()
        datafromxml = xml_to_dict(root)
        self._ntraj = datafromxml["ntraj"]
        for n in range(self._ntraj):
            trajId = formTrajID(n)
            T_entry = root.find(trajId)
            if T_entry is not None:
                chkFile = T_entry.text
                if os.path.exists(chkFile):
                    nTrajRestored += 1
                    self._trajs_db.append(
                        Trajectory.restoreFromChk(
                            chkFile,
                            fmodel_t=self._fmodel_t,
                            parameters=self._parameters
                        )
                    )
                else:
                    raise DatabaseError(
                        "Could not find the trajectory checkFile {} listed in the TAMS database !".format(
                            chkFile
                        )
                    )
            else:
                self._trajs_db.append(
                    Trajectory(
                        fmodel_t=self._fmodel_t,
                        parameters=self._parameters,
                        trajId=formTrajID(n),
                    )
                )
        return nTrajRestored


    def _check_database_consistency(self) -> None:
        """Check the restart database consistency."""
        # Open and load header
        headerFile = "{}/header.xml".format(self._name)
        tree = ET.parse(headerFile)
        root = tree.getroot()
        headerfromxml = xml_to_dict(root.find("metadata"))
        if self._fmodel_t.name() != headerfromxml["model_t"]:
            raise DatabaseError(
                "Trying to restore a TAMS with {} model from database with {} model !".format(
                    self._fmodel_t.name(), headerfromxml["model_t"]
                )
            )

        # Parameters stored in the DB override
        # newly provided parameters.
        with open("{}/input_params.toml".format(self._name), 'r') as f:
            readInParams = toml.load(f)
        self._parameters.update(readInParams)

    def appendTraj(self, a_traj: Trajectory) -> None:
        """Append a Trajectory to the internal list."""
        self._trajs_db.append(a_traj)

    def trajList(self) -> list[Trajectory]:
        """Access to the trajectory list."""
        return self._trajs_db

    def getTraj(self, idx: int) -> Trajectory:
        """Access to a given trajectory."""
        assert(idx < len(self._trajs_db))
        return self._trajs_db[idx]

    def overwriteTraj(self,
                      idx: int,
                      traj: Trajectory) -> None:
        """Deep copy a trajectory into internal list."""
        self._trajs_db[idx] = copy.deepcopy(traj)
        if self._save:
            tid = self._trajs_db[idx].id()
            self._trajs_db[idx].setCheckFile("{}/{}/{}.xml".format(self._name, "trajectories", tid))
            self._trajs_db[idx].store()

    def isEmpty(self) -> bool:
        """Check if database is empty."""
        return self.trajListLen() == 0

    def trajListLen(self) -> int:
        """Length of the trajectory list."""
        return len(self._trajs_db)

    def updateTrajList(self, a_trajList: list[Trajectory]) -> None:
        """Overwrite the trajectory list."""
        self._trajs_db = a_trajList

    def updateDiskData(self) -> None:
        """Update trajectory list stored to disk."""
        self.appendTrajsToDB()

    def weights(self) -> list[float]:
        """Splitting iterations weights."""
        return  self._weights

    def appendWeight(self, weight: float) -> None:
        """Append a weight to internal list."""
        self._weights.append(weight)

    def biases(self) -> list[int]:
        """Splitting iterations biases."""
        return self._l_bias

    def appendBias(self, bias: int) -> None:
        """Append a bias to internal list."""
        self._l_bias.append(bias)

    def kSplit(self) -> int:
        """Splitting iteration counter."""
        return self._ksplit

    def setKSplit(self, ksplit: int) -> None:
        """Set splitting iteration counter."""
        self._ksplit = ksplit

    def countEndedTraj(self) -> int:
        """Return the number of trajectories that ended."""
        count = 0
        for T in self._trajs_db:
            if T.hasEnded():
                count = count + 1
        return count

    def countConvergedTraj(self) -> int:
        """Return the number of trajectories that converged."""
        count = 0
        for T in self._trajs_db:
            if T.isConverged():
                count = count + 1
        return count

    def getTransitionProbability(self) -> float:
        """Return the transition probability."""
        if self.countEndedTraj() < self._ntraj:
            print("TAMS initialization still ongoing, probability estimate not available yet")
            return 0.0
        else:
            W = self._ntraj * self._weights[-1]
            for i in range(len(self._l_bias)):
                W += self._l_bias[i] * self._weights[i]

            trans_prob = self.countConvergedTraj() * self._weights[-1] / W
            return trans_prob


    def info(self):
        """Print database info to screen."""
        version, db_date, db_model = self._readHeader()
        print("################################################")
        print("# TAMS v{:13s} trajectory database      #".format(version))
        print("# Date: {:38s} #".format(str(db_date)))
        print("# Model: {:37s} #".format(db_model))
        print("################################################")
        print("# Requested # of traj: {:23} #".format(self._ntraj))
        print("# Requested # of splitting iter: {:13} #".format(self._nsplititer))
        print("# Number of 'Ended' trajectories: {:12} #".format(self.countEndedTraj()))
        print("# Number of 'Converged' trajectories: {:8} #".format(self.countConvergedTraj()))
        print("# Current splitting iter counter: {:12} #".format(self._ksplit))
        if self.countEndedTraj() < self._ntraj:
            print("# Transition probability: {:24} #".format(self.getTransitionProbability()))
        print("################################################")

    def plotScoreFunctions(self, fname: str = None) -> None:
        """Plot the score as function of time for all trajectories."""
        if not fname:
            pltfile = Path(self._name).stem + "_scores.png"
        else:
            pltfile = fname
        plt.figure(figsize=(10, 6))
        for T in self._trajs_db:
            plt.plot(T.getTimeArr(), T.getScoreArr(), linewidth=0.8)

        plt.xlabel(r'$Time$', fontsize="x-large")
        plt.xlim(left=0.0)
        plt.ylabel(r'$Score \; [-]$', fontsize="x-large")
        plt.xticks(fontsize="x-large")
        plt.yticks(fontsize="x-large")
        plt.grid(linestyle='dotted')
        plt.tight_layout() # to fit everything in the prescribed area
        plt.savefig(fname, dpi=300)
        plt.clf()
