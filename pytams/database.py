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
from pytams.trajectory import Trajectory
from pytams.trajectory import formTrajID
from pytams.xmlutils import new_element
from pytams.xmlutils import xml_to_dict

_logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception class for TAMS Database."""

    pass


class Database:
    """A database class for TAMS."""

    def __init__(self,
                 fmodel_t: Any,
                 params: dict,
                 ntraj: Optional[int] = None,
                 nsplititer: Optional[int] = None) -> None:
        """Init a TAMS database.

        Args:
            fmodel_t: the forward model type
            params: a dictionary of parameters
            ntraj: [OPT] number of traj to hold
            nsplititer: [OPT] number of splitting iteration to hold
        """
        self._fmodel_t = fmodel_t

        # Metadata
        self._verbose = False
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
        self._name = "{}.tdb".format(self._prefix)

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
                warn_msg = f"Database {self._name} already present. It will be copied to {name_rnd}"
                _logger.warning(warn_msg)
                shutil.move(self._name, name_rnd)

            os.mkdir(self._name)

            # Save the runtime options
            with open("{}/input_params.toml".format(self._name), 'w') as f:
                toml.dump(self._parameters, f)

            # Header file with metadata
            self._writeMetadata()

            # Empty trajectories subfolder
            os.mkdir("{}/{}".format(self._name, "trajectories"))

    def _writeMetadata(self) -> None:
        """Write the database Metadata to disk."""
        if self._format == "XML":
            headerFile = self.headerFile()
            root = ET.Element("header")
            mdata = ET.SubElement(root, "metadata")
            mdata.append(new_element("pyTAMS_version", version(__package__)))
            mdata.append(new_element("date", datetime.now()))
            mdata.append(new_element("model_t", self._fmodel_t.name()))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(headerFile)

            # Initialialize splitting data file
            self.saveSplittingData()

            # Dynamically updated file with trajectory pool
            # Empty for now
            databaseFile = self.poolFile()
            root = ET.Element("trajectories")
            root.append(new_element("ntraj", self._ntraj))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)
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


    def appendTrajsToDB(self) -> None:
        """Append started trajectories to the pool file."""
        if self._save:
            inf_msg = f"Appending started trajectories to database {self._name}"
            _logger.info(inf_msg)
            databaseFile = self.poolFile()
            tree = ET.parse(databaseFile)
            root = tree.getroot()
            for T in self._trajs_db:
                T_entry = root.find(T.idstr())
                if T.hasStarted() and T_entry is None:
                    loc = T.checkFile()
                    root.append(new_element(T.idstr(), loc))

            ET.indent(tree, space="\t", level=0)
            tree.write(databaseFile)

    def saveSplittingData(self,
                          ongoing_trajs: Optional[list[int]] = None) -> None:
        """Write splitting data."""
        if not self._save:
            return

        # Splitting data file
        if self._format == "XML":
            splittingDataFile = "{}/splittingData.xml".format(self._name)
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
            splittingDataFile = "{}/splittingData.xml".format(self._name)
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
            dbFile = self.poolFile()
            nTrajRestored = self.loadTrajectoryDB(dbFile)

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
                assert(chkFile is not None)
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
                    err_msg = f"Could not find the trajectory checkFile {chkFile} listed in the TAMS database !"
                    _logger.error(err_msg)
                    raise DatabaseError(err_msg)
            else:
                self._trajs_db.append(
                    Trajectory(
                        fmodel_t=self._fmodel_t,
                        parameters=self._parameters,
                        trajId=n,
                    )
                )
        return nTrajRestored


    def _check_database_consistency(self) -> None:
        """Check the restart database consistency."""
        # Open and load header
        tree = ET.parse(self.headerFile())
        root = tree.getroot()
        headerfromxml = xml_to_dict(root.find("metadata"))
        if self._fmodel_t.name() != headerfromxml["model_t"]:
            err_msg = f"Trying to restore a TAMS with {self._fmodel_t.name()}"\
                      f"model from database with {headerfromxml['model_t']} model !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

        # Parameters stored in the DB override
        # newly provided parameters.
        with open("{}/input_params.toml".format(self._name), 'r') as f:
            readInParams = toml.load(f)
        self._parameters.update(readInParams)

    def name(self) -> str:
        """Accessor to DB name."""
        return self._name

    def save(self) -> bool:
        """Accessor to DB save bool."""
        return self._save

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

    def headerFile(self) -> str:
        """Helper returning the DB header file."""
        return "{}/header.xml".format(self._name)

    def poolFile(self) -> str:
        """Helper returning the DB trajectory pool file."""
        return "{}/trajPool.xml".format(self._name)

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
