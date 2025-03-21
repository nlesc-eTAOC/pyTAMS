"""A database class for TAMS."""
import copy
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import toml
from pytams.sqldb import SQLFile
from pytams.trajectory import Trajectory
from pytams.xmlutils import new_element
from pytams.xmlutils import xml_to_dict

_logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception class for TAMS Database."""

    pass


class Database:
    """A database class for TAMS.

    The database class for TAMS is a container for
    all the trajectory and splitting data. When the
    user requires to store the database, a local folder is
    created holding a number of readable files, any output
    from the model and an SQL file used to lock/release
    trajectories as the TAMS algorithm proceeds.

    The readable files are currently in an XML format.

    A database can be loaded independently from the TAMS
    algorithm and used for post-processing.

    Attributes:
        _fmodel_t: the forward model type
        _save: boolean to trigger saving the database to disk
        _load: a path to an existing database to restore
        _parameters: the dictionary of parameters
        _trajs_db: the list of trajectories
        _ksplit: the current splitting iteration
        _l_bias: the list of bias
        _weights: the list of weights
        _ongoing: the list of ongoing branches if unfinished splitting iteration.
    """

    def __init__(self,
                 fmodel_t: Any,
                 params: dict,
                 ntraj: Optional[int] = None,
                 nsplititer: Optional[int] = None) -> None:
        """Initialize a TAMS database.

        Initialize in-memory TAMS database, loading data
        from an existing database if provided.

        To prevent overriding an existing database by mistake,
        if a database folder with the same name already exists,
        if will be copied to a new folder with a random name unless
        that same folder is also specified as a restart database.

        Args:
            fmodel_t: the forward model type
            params: a dictionary of parameters
            ntraj: [OPT] number of traj to hold
            nsplititer: [OPT] number of splitting iteration to hold
        """
        self._fmodel_t = fmodel_t

        # Metadata
        self._save = False
        self._load : Union[str,None] = None
        self._parameters = params

        # Trajectory Pool
        self._trajs_db : list[Trajectory] = []

        # Splitting data
        self._ksplit = 0
        self._l_bias : list[int] = []
        self._weights : list[float] = [1.0]
        self._ongoing = None

        self._save = params.get("database", {}).get("DB_save", False)
        self._prefix = params.get("database", {}).get("DB_prefix", "TAMS")
        self._load = params.get("database", {}).get("DB_restart", None)
        self._format = params.get("database", {}).get("DB_format", "XML")
        self._name = f"{self._prefix}.tdb"

        if self._load:
            self._name = self._load
            self._restoreTrajDB()
        else:
            if not ntraj:
                err_msg = "Initializing TAMS database from scratch require ntraj !"
                _logger.error(err_msg)
                raise DatabaseError(err_msg)
            if not nsplititer:
                err_msg = "Initializing TAMS database from scratch require nsplititer !"
                _logger.error(err_msg)
                raise DatabaseError(err_msg)
            self._ntraj = ntraj
            self._nsplititer = nsplititer
            self._setUpTree()

    def nTraj(self) -> int:
        """Return the number of trajectory used for TAMS.

        Note that this is the requested number of trajectory, not
        the current length of the trajectory pool.

        Return:
            number of trajectory
        """
        return self._ntraj

    def nSplitIter(self) -> int:
        """Return the number of splitting iteration used for TAMS.

        Note that this is the requested number of splitting iteration, not
        the current splitting iteration.

        Return:
            number of splitting iteration
        """
        return self._nsplititer

    def _setUpTree(self) -> None:
        """Initialize the trajectory database tree."""
        if self._save:
            if os.path.exists(self._name) and self._name != self._load:
                rng = np.random.default_rng(12345)
                copy_exists = True
                while copy_exists:
                    random_int = rng.integers(0, 999999)
                    name_rnd = f"{self._name}_{random_int:06d}"
                    copy_exists = os.path.exists(name_rnd)
                warn_msg = f"Database {self._name} already present. It will be copied to {name_rnd}"
                _logger.warning(warn_msg)
                shutil.move(self._name, name_rnd)

            os.mkdir(self._name)

            # Save the runtime options
            with open(f"{self._name}/input_params.toml", 'w') as f:
                toml.dump(self._parameters, f)

            # Header file with metadata and pool DB
            self._writeMetadata()

            # Empty trajectories subfolder
            os.mkdir(f"{self._name}/trajectories")

    def _writeMetadata(self) -> None:
        """Write the database Metadata to disk."""
        if self._format == "XML":
            headerFile = self.headerFile()
            root = ET.Element("header")
            mdata = ET.SubElement(root, "metadata")
            mdata.append(new_element("pyTAMS_version", version(__package__)))
            mdata.append(new_element("date", datetime.now()))
            mdata.append(new_element("model_t", self._fmodel_t.name()))
            mdata.append(new_element("ntraj", self._ntraj))
            mdata.append(new_element("nsplititer", self._nsplititer))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(headerFile)

            # Initialialize splitting data file
            self.saveSplittingData()

            # Initialize the SQL pool file
            self._pool_db = SQLFile(self.poolFile())
        else:
            err_msg = f"Unsupported TAMS database format: {self._format} !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

    def _readHeader(self) -> tuple[str, datetime, str]:
        """Read the database Metadata to header."""
        if self._load:
            if self._format == "XML":
                tree = ET.parse(self.headerFile())
                root = tree.getroot()
                mdata = root.find("metadata")
                datafromxml = xml_to_dict(mdata)
                pyTAMS_version = datafromxml["pyTAMS_version"]
                db_date = datafromxml["date"]
                db_model = datafromxml["model_t"]

                return pyTAMS_version, db_date, db_model
            else:
                err_msg = f"Unsupported TAMS database format: {self._format} !"
                _logger.error(err_msg)
                raise DatabaseError(err_msg)
        else:
            return "Error", datetime.min, "Error"

    def saveSplittingData(self,
                          ongoing_trajs: Optional[list[int]] = None) -> None:
        """Write splitting data to the database.

        Args:
            ongoing_trajs: an optional list of ongoing trajectories
        """
        if not self._save:
            return

        # Splitting data file
        if self._format == "XML":
            splittingDataFile = f"{self._name}/splittingData.xml"
            root = ET.Element("splitting")
            root.append(new_element("nsplititer", self._nsplititer))
            root.append(new_element("ksplit", self._ksplit))
            root.append(new_element("bias", np.array(self._l_bias, dtype=int)))
            root.append(new_element("weight", np.array(self._weights, dtype=float)))
            if ongoing_trajs:
                root.append(new_element("ongoing", np.array(ongoing_trajs)))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(splittingDataFile)
        else:
            err_msg = f"Unsupported TAMS database format: {self._format} !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

    def _readSplittingData(self) -> None:
        """Read splitting data."""
        # Read data file
        if self._format == "XML":
            splittingDataFile = f"{self._name}/splittingData.xml"
            tree = ET.parse(splittingDataFile)
            root = tree.getroot()
            datafromxml = xml_to_dict(root)
            self._nsplititer = datafromxml["nsplititer"]
            self._ksplit = datafromxml["ksplit"]
            self._l_bias = datafromxml["bias"].tolist()
            self._weights = datafromxml["weight"].tolist()
            if "ongoing" in datafromxml:
                self._ongoing = datafromxml["ongoing"].tolist()
        else:
            err_msg = f"Unsupported TAMS database format: {self._format} !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

    def _restoreTrajDB(self) -> None:
        """Initialize TAMS from a stored trajectory database."""
        assert(self._load is not None)
        if os.path.exists(self._load):
            inf_msg = f"Loading TAMS database: {self._load}"
            _logger.info(inf_msg)

            # Check the database parameters against current run
            self._check_database_consistency()

            # Load splitting data
            self._readSplittingData()

            # Load trajectories stored in the database when available.
            nTrajRestored = self.loadTrajectoryDB(self.poolFile())

            inf_msg = f"{nTrajRestored} trajectories loaded"
            _logger.info(inf_msg)
        else:
            err_msg = f"Could not find the {self._load} TAMS database !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

    def loadTrajectoryDB(self, dbFile: str) -> int:
        """Load trajectories stored into the database.

        Args:
            dbFile: the database file

        Return:
            number of trajectories loaded
        """
        db_handle = SQLFile(dbFile)
        # Counter for number of trajectory loaded
        nTrajRestored = 0

        ntraj_in_db = db_handle.get_trajectory_count()
        for n in range(ntraj_in_db):
            trajChkFile = db_handle.fetch_trajectory(n)
            if os.path.exists(trajChkFile):
                nTrajRestored += 1
                self._trajs_db.append(
                    Trajectory.restoreFromChk(
                        trajChkFile,
                        fmodel_t=self._fmodel_t,
                        parameters=self._parameters
                    )
                )
            else:
                T = Trajectory(
                    fmodel_t=self._fmodel_t,
                    parameters=self.parameters,
                    trajId=n,
                )
                T.setCheckFile(trajChkFile)
                self._trajs_db.append(T)

        return nTrajRestored


    def _check_database_consistency(self) -> None:
        """Check the restart database consistency.

        Perform some basic checks on the consistency between the
        input file and the database being loaded.

        Return:
            DatabaseError if the consistency check fails
        """
        # Open and load header
        tree = ET.parse(self.headerFile())
        root = tree.getroot()
        headerfromxml = xml_to_dict(root.find("metadata"))
        if self._fmodel_t.name() != headerfromxml["model_t"]:
            err_msg = f"Trying to restore a TAMS with {self._fmodel_t.name()}"\
                      f"model from database with {headerfromxml['model_t']} model !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

        self._ntraj = headerfromxml["ntraj"]
        self._nsplititer = headerfromxml["nsplititer"]

        # Parameters stored in the DB override
        # newly provided parameters.
        with open(f"{self._load}/input_params.toml", 'r') as f:
            readInParams = toml.load(f)
        self._parameters.update(readInParams)

    def name(self) -> str:
        """Accessor to DB name.

        Return:
            DB name
        """
        return self._name

    def save(self) -> bool:
        """Accessor to DB save bool.

        Return:
            Save bool
        """
        return self._save

    def appendTraj(self, a_traj: Trajectory) -> None:
        """Append a Trajectory to the internal list.

        Args:
            a_traj: the trajectory
        """
        self._trajs_db.append(a_traj)

        # Also adds it to the SQL pool file.
        if self._save:
            self._pool_db.add_trajectory(a_traj.checkFile())

    def trajList(self) -> list[Trajectory]:
        """Access to the trajectory list.

        Return:
            Trajectory list
        """
        return self._trajs_db

    def getTraj(self, idx: int) -> Trajectory:
        """Access to a given trajectory.

        Args:
            idx: the index

        Return:
            Trajectory

        Raises:
            ValueError if idx is out of range
        """
        if (idx < 0 or
            idx >= len(self._trajs_db)):
            err_msg = f"Trying to access a non existing trajectory {idx} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        return self._trajs_db[idx]

    def overwriteTraj(self,
                      idx: int,
                      traj: Trajectory) -> None:
        """Deep copy a trajectory into internal list.

        Args:
            idx: the index of the trajectory to override
            traj: the new trajectory

        Raises:
            ValueError if idx is out of range
        """
        if (idx < 0 or
            idx >= len(self._trajs_db)):
            err_msg = f"Trying to override a non existing trajectory {idx} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        self._trajs_db[idx] = copy.deepcopy(traj)

    def headerFile(self) -> str:
        """Helper returning the DB header file.

        Return:
            Header file
        """
        return f"{self._name}/header.xml"

    def poolFile(self) -> str:
        """Helper returning the DB trajectory pool file.

        Return:
            Pool file
        """
        return f"{self._name}/trajPool.db"

    def isEmpty(self) -> bool:
        """Check if list of trajectories is empty.

        Return:
            True if the list of trajectories is empty
        """
        return self.trajListLen() == 0

    def trajListLen(self) -> int:
        """Length of the trajectory list.

        Return:
            Trajectory list length
        """
        return len(self._trajs_db)

    def updateTrajList(self,
                       a_trajList: list[Trajectory]) -> None:
        """Overwrite the trajectory list.

        Args:
            a_trajList: the new trajectory list
        """
        self._trajs_db = a_trajList

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

    def reset_ongoing(self) -> None:
        """Reset the list of trajectories undergoing branching."""
        self._ongoing = None

    def get_ongoing(self) -> Union[list[int],None]:
        """Return the list of trajectories undergoing branching or None."""
        return self._ongoing

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


    def info(self) -> None:
        """Print database info to screen."""
        version, db_date, db_model = self._readHeader()
        db_date_str = str(db_date)
        prettyLine = "####################################################"
        inf_tbl = f"""
            {prettyLine}
            # TAMS v{version:17s} trajectory database      #
            # Date: {db_date_str:42s} #
            # Model: {db_model:41s} #
            {prettyLine}
            # Requested # of traj: {self._ntraj:27} #
            # Requested # of splitting iter: {self._nsplititer:17} #
            # Number of 'Ended' trajectories: {self.countEndedTraj():16} #
            # Number of 'Converged' trajectories: {self.countConvergedTraj():12} #
            # Current splitting iter counter: {self._ksplit:16} #
            # Transition probability: {self.getTransitionProbability():24} #
            {prettyLine}
        """
        print(inf_tbl)

    def plotScoreFunctions(self,
                           fname: Optional[str] = None) -> None:
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
        plt.savefig(pltfile, dpi=300)
        plt.clf()
        plt.close()
