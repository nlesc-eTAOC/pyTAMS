"""A database class for TAMS."""
from __future__ import annotations
import copy
import logging
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union
import cloudpickle
import dill
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
    user provides a path to store the database, a local folder is
    created holding a number of readable files, any output
    from the model and an SQL file used to lock/release
    trajectories as the TAMS algorithm proceeds.

    The readable files are currently in an XML format.

    A database can be loaded independently from the TAMS
    algorithm and used for post-processing.

    Attributes:
        _fmodel_t: the forward model type
        _save: boolean to trigger saving the database to disk
        _path: a path to an existing database to restore or a new path
        _restart: a bool to override an existing database
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

        Initialize TAMS database object, bare in-memory or on-disk.

        On-disk database trigger if a path is provided in the
        parameters dictonary. The user can chose to not append/override
        the existing database in which case the existing path
        will be copied to a new random name.

        Args:
            fmodel_t: the forward model type
            params: a dictionary of parameters
            ntraj: [OPT] number of traj to hold
            nsplititer: [OPT] number of splitting iteration to hold
        """
        self._fmodel_t = fmodel_t

        # Metadata
        self._save = False
        self._parameters = params
        self._name = "TAMS_" + fmodel_t.name()
        self._path = params.get("database", {}).get("path", None)
        if self._path:
            self._save = True
            self._restart = params.get("database", {}).get("restart", False)
            self._format = params.get("database", {}).get("format", "XML")
            if self._format not in ["XML"]:
                err_msg = f"Unsupported TAMS database format: {self._format} !"
                _logger.error(err_msg)
                raise DatabaseError(err_msg)
            self._name = f"{self._path}"
            self._abs_path : os.PathLike = Path.cwd() / self._name
            self._creation_date = datetime.now()
            self._version = version(__package__)

        # Trajectory Pool
        self._trajs_db : list[Trajectory] = []
        self._pool_db = None

        # Splitting data
        self._ksplit = 0
        self._l_bias : list[int] = []
        self._weights : list[float] = [1.0]
        self._ongoing = None

        # Initialize only metadata at this point
        # so that the object remains lightweight
        self._init_metadata(ntraj, nsplititer)

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

    def path(self) -> str:
        """Return the path to the database."""
        return self._path

    @classmethod
    def load(cls,
             a_path: os.PathLike) -> Database:
        """Instanciate a TAMS database from disk.

        Args:
            a_path: the path to the database

        Return:
            a TAMS database object
        """
        if not a_path.exists():
            err_msg = f"Database {a_path} does not exist !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

        # Load necessary elements to call the constructor
        # Ensure that the database is not restarted at this step
        db_params = toml.load(a_path / "input_params.toml")
        db_params["database"].update({"restart": False})

        # If the a_path differs from the one stored in the
        # database (the DB has been moved), update the path
        if str(a_path) != db_params["database"]["path"]:
            warn_msg = f"Database {db_params['database']['path']} has been moved to {a_path} !"
            _logger.warning(warn_msg)
            db_params["database"]["path"] = str(a_path)
        model_file = Path(a_path / "fmodel.pkl")
        with model_file.open("rb") as f:
            model = dill.load(f)

        return cls(model, db_params)

    def _init_metadata(self,
                       ntraj: Optional[int] = None,
                       nsplititer: Optional[int] = None) -> None:
        """Initialize the database.

        Initialize database internal metadata (only) and setup
        the database on disk if needed.

        Args:
            ntraj: [OPT] number of traj to hold
            nsplititer: [OPT] number of splitting iteration to hold
        """
        # Initialize or load disk-based database metadata
        if self._save:
            # Check for an existing database:
            db_exists = self._abs_path.exists()

            # Regardless of a pre-existing path we initialize from scratch
            if (not db_exists or self._restart):
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
                self._setup_tree()

            # Load the database
            else:
                self._load_metadata()
                # Parameters stored in the DB override
                # newly provided parameters.
                with Path(self._abs_path / "input_params.toml").open('r') as f:
                    readInParams = toml.load(f)
                self._parameters.update(readInParams)

        # Initialize in-memory database metadata
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


    def _setup_tree(self) -> None:
        """Initialize the trajectory database tree."""
        if self._save:
            if self._abs_path.exists():
                rng = np.random.default_rng(12345)
                copy_exists = True
                while copy_exists:
                    random_int = rng.integers(0, 999999)
                    path_rnd = Path.cwd() / f"{self._name}_{random_int:06d}"
                    copy_exists = path_rnd.exists()
                warn_msg = f"Database {self._name} already present. It will be copied to {path_rnd.name}"
                _logger.warning(warn_msg)
                shutil.move(self._name, path_rnd.absolute())

            os.mkdir(self._name)

            # Save the runtime options
            with Path(self._abs_path / "input_params.toml").open('w') as f:
                toml.dump(self._parameters, f)

            # Header file with metadata and pool DB
            self._write_metadata()

            # Serialize the model
            model_file = Path(self._abs_path / "fmodel.pkl")
            cloudpickle.register_pickle_by_value(sys.modules[self._fmodel_t.__module__])
            with model_file.open("wb") as f:
                cloudpickle.dump(self._fmodel_t, f)

            # Empty trajectories subfolder
            Path(self._abs_path / "trajectories").mkdir(parents=True)

    def _write_metadata(self) -> None:
        """Write the database Metadata to disk."""
        if self._format == "XML":
            headerFile = self.header_file()
            root = ET.Element("header")
            mdata = ET.SubElement(root, "metadata")
            mdata.append(new_element("pyTAMS_version", version(__package__)))
            mdata.append(new_element("date", self._creation_date))
            mdata.append(new_element("model_t", self._fmodel_t.name()))
            mdata.append(new_element("ntraj", self._ntraj))
            mdata.append(new_element("nsplititer", self._nsplititer))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(headerFile)

            # Initialialize splitting data file
            self.save_splitting_data()

            # Initialize the SQL pool file
            self._pool_db = SQLFile(self.pool_file())
        else:
            err_msg = f"Unsupported TAMS database format: {self._format} !"
            _logger.error(err_msg)
            raise DatabaseError(err_msg)

    def _load_metadata(self) -> None:
        """Read the database Metadata from the header."""
        if self._save:
            if self._format == "XML":
                tree = ET.parse(self.header_file())
                root = tree.getroot()
                mdata = root.find("metadata")
                datafromxml = xml_to_dict(mdata)
                self._ntraj = datafromxml["ntraj"]
                self._nsplititer = datafromxml["nsplititer"]
                self._version = datafromxml["pyTAMS_version"]
                if self._version != version(__package__):
                    warn_msg = f"Database pyTAMS version {self._version} is different from {version(__package__)}"
                    _logger.warning(warn_msg)
                self._creation_date = datafromxml["date"]
                db_model = datafromxml["model_t"]
                if db_model != self._fmodel_t.name():
                    err_msg = f"Database model {db_model} is different from call {self._fmodel_t.name()}"
                    _logger.error(err_msg)
                    raise DatabaseError(err_msg)

                # Initialize the SQL pool file
                self._pool_db = SQLFile(self.pool_file())
            else:
                err_msg = f"Unsupported TAMS database format: {self._format} !"
                _logger.error(err_msg)
                raise DatabaseError(err_msg)


    def save_trajectory(self, traj: Trajectory) -> None:
        """Save a trajectory to disk in the database.

        Args:
            traj: the trajectory to save
        """
        if not self._save:
            return

        traj.store()

    def save_splitting_data(self,
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

    def _read_splitting_data(self) -> None:
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

    def load_data(self) -> None:
        """Load data stored into the database."""
        if not self._save:
            return

        # Counter for number of trajectory loaded
        nTrajRestored = 0

        ntraj_in_db = self._pool_db.get_trajectory_count()
        for n in range(ntraj_in_db):
            trajChkFile = Path(self._abs_path) / self._pool_db.fetch_trajectory(n)
            if trajChkFile.exists():
                nTrajRestored += 1
                self._trajs_db.append(
                    Trajectory.restoreFromChk(
                        trajChkFile,
                        fmodel_t=self._fmodel_t,
                        parameters=self._parameters
                    )
                )
                #self._trajs_db[-1].setCheckFile(trajChkFile.relative_to(self._abs_path))
                self._trajs_db[-1].setCheckFile(trajChkFile)
            else:
                T = Trajectory(
                    fmodel_t=self._fmodel_t,
                    parameters=self._parameters,
                    trajId=n,
                )
                #T.setCheckFile(trajChkFile.relative_to(self._abs_path))
                T.setCheckFile(trajChkFile)
                self._trajs_db.append(T)

        inf_msg = f"{nTrajRestored} trajectories loaded"
        _logger.info(inf_msg)

        # Load splitting data
        self._read_splitting_data()

        self.info()

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

    def append_traj(self, a_traj: Trajectory) -> None:
        """Append a Trajectory to the internal list.

        Args:
            a_traj: the trajectory
        """
        # Also adds it to the SQL pool file.
        # and set the checkfile
        if self._save:
            checkfile_str = f"./trajectories/{a_traj.idstr()}.xml"
            checkfile = Path(self._abs_path) / checkfile_str
            a_traj.setCheckFile(checkfile)
            self._pool_db.add_trajectory(checkfile_str)

        self._trajs_db.append(a_traj)


    def traj_list(self) -> list[Trajectory]:
        """Access to the trajectory list.

        Return:
            Trajectory list
        """
        return self._trajs_db

    def get_traj(self, idx: int) -> Trajectory:
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

    def overwrite_traj(self,
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

    def header_file(self) -> str:
        """Helper returning the DB header file.

        Return:
            Header file
        """
        return f"{self._name}/header.xml"

    def pool_file(self) -> str:
        """Helper returning the DB trajectory pool file.

        Return:
            Pool file
        """
        return f"{self._name}/trajPool.db"

    def is_empty(self) -> bool:
        """Check if list of trajectories is empty.

        Return:
            True if the list of trajectories is empty
        """
        return self.traj_list_len() == 0

    def traj_list_len(self) -> int:
        """Length of the trajectory list.

        Return:
            Trajectory list length
        """
        return len(self._trajs_db)

    def update_traj_list(self,
                         a_trajList: list[Trajectory]) -> None:
        """Overwrite the trajectory list.

        Args:
            a_trajList: the new trajectory list
        """
        self._trajs_db = a_trajList

    def lock_trajectory(self,
                        tid: int,
                        allow_completed_lock : bool = False) -> bool:
        """Lock a trajectory in the SQL DB.

        Args:
            tid: the trajectory id
            allow_completed_lock: True if the trajectory can be locked even if it is completed

        Return:
            True if no disk DB and the trajectory was locked
        """
        if not self._save:
            return True
        return self._pool_db.lock_trajectory(tid, allow_completed_lock)

    def unlock_trajectory(self,
                          tid: int,
                          hasEnded: bool) -> None:
        """Unlock a trajectory in the SQL DB.

        Args:
            tid: the trajectory id
            hasEnded: True if the trajectory has ended
        """
        if not self._save:
            return

        if hasEnded:
            self._pool_db.mark_trajectory_as_completed(tid)
        else:
            self._pool_db.release_trajectory(tid)


    def weights(self) -> list[float]:
        """Splitting iterations weights."""
        return  self._weights

    def append_weight(self, weight: float) -> None:
        """Append a weight to internal list."""
        self._weights.append(weight)

    def biases(self) -> list[int]:
        """Splitting iterations biases."""
        return self._l_bias

    def append_bias(self, bias: int) -> None:
        """Append a bias to internal list."""
        self._l_bias.append(bias)

    def kSplit(self) -> int:
        """Splitting iteration counter."""
        return self._ksplit

    def done_with_splitting(self) -> bool:
        """Check if we are done with splitting."""
        return self._ksplit >= self._nsplititer

    def reset_ongoing(self) -> None:
        """Reset the list of trajectories undergoing branching."""
        self._ongoing = None

    def get_ongoing(self) -> Union[list[int],None]:
        """Return the list of trajectories undergoing branching or None."""
        return self._ongoing

    def setKSplit(self, ksplit: int) -> None:
        """Set splitting iteration counter."""
        self._ksplit = ksplit

    def count_ended_traj(self) -> int:
        """Return the number of trajectories that ended."""
        count = 0
        for T in self._trajs_db:
            if T.hasEnded():
                count = count + 1
        return count

    def count_converged_traj(self) -> int:
        """Return the number of trajectories that converged."""
        count = 0
        for T in self._trajs_db:
            if T.isConverged():
                count = count + 1
        return count

    def get_transition_probability(self) -> float:
        """Return the transition probability."""
        if self.count_ended_traj() < self._ntraj:
            print("TAMS initialization still ongoing, probability estimate not available yet")
            return 0.0
        else:
            W = self._ntraj * self._weights[-1]
            for i in range(len(self._l_bias)):
                W += self._l_bias[i] * self._weights[i]

            trans_prob = self.count_converged_traj() * self._weights[-1] / W
            return trans_prob


    def info(self) -> None:
        """Print database info to screen."""
        db_date_str = str(self._creation_date)
        prettyLine = "####################################################"
        inf_tbl = f"""
            {prettyLine}
            # TAMS v{self._version:17s} trajectory database      #
            # Date: {db_date_str:42s} #
            # Model: {self._fmodel_t.name():41s} #
            {prettyLine}
            # Requested # of traj: {self._ntraj:27} #
            # Requested # of splitting iter: {self._nsplititer:17} #
            # Number of 'Ended' trajectories: {self.count_ended_traj():16} #
            # Number of 'Converged' trajectories: {self.count_converged_traj():12} #
            # Current splitting iter counter: {self._ksplit:16} #
            # Transition probability: {self.get_transition_probability():24} #
            {prettyLine}
        """
        print(inf_tbl)

    def plot_score_functions(self,
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
