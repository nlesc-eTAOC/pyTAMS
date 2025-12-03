"""A database class for TAMS."""

from __future__ import annotations
import copy
import datetime
import logging
import shutil
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import toml
from pytams.sqldb import SQLFile
from pytams.trajectory import Trajectory
from pytams.trajectory import form_trajectory_id
from pytams.utils import get_module_local_import
from pytams.xmlutils import new_element
from pytams.xmlutils import xml_to_dict

if TYPE_CHECKING:
    from pytams.fmodel import ForwardModelBaseClass

_logger = logging.getLogger(__name__)


class Database:
    """A database class for TAMS.

    The database class for TAMS is a container for
    all the trajectory and splitting data. When the
    user provides a path to store the database, a local folder is
    created holding a number of readable files, any output
    from the model and SQL files used to lock/release
    trajectories as the TAMS algorithm proceeds.

    The readable files are currently in an XML format.

    A database can be loaded independently from the TAMS
    algorithm and used for post-processing.

    Attributes:
        _fmodel_t: the forward model type
        _save_to_disk: boolean to trigger saving the database to disk
        _path: a path to an existing database to restore or a new path
        _restart: a bool to override an existing database
        _parameters: the dictionary of parameters
        _trajs_db: the list of trajectories
    """

    def __init__(
        self,
        fmodel_t: type[ForwardModelBaseClass],
        params: dict[Any, Any],
        ntraj: int = -1,
        nsplititer: int = -1,
        read_only: bool = True,
    ) -> None:
        """Initialize a TAMS database.

        Initialize TAMS database object, bare in-memory or on-disk.

        On-disk database trigger if a path is provided in the
        parameters dictionary. The user can chose to not append/override
        the existing database in which case the existing path
        will be copied to a new random name.

        Args:
            fmodel_t: the forward model type
            params: a dictionary of parameters
            ntraj: [OPT] number of traj to hold
            nsplititer: [OPT] number of splitting iteration to hold
            read_only: [OPT] boolean setting database access mode
        """
        # Access mode
        self._read_only = read_only

        # For posterity
        self._creation_date = datetime.datetime.now(tz=datetime.timezone.utc)
        self._version = version(__package__)
        self._name = "TAMS_" + fmodel_t.name()

        # Stash away the model class and parameters
        self._fmodel_t = fmodel_t
        self._parameters = params

        # Database format/storage parameters
        self._save_to_disk = False
        self._path: str | None = params.get("database", {}).get("path", None)
        if self._path:
            self._save_to_disk = True
            self._restart = params.get("database", {}).get("restart", False)
            self._format = params.get("database", {}).get("format", "XML")
            if self._format not in ["XML"]:
                err_msg = f"Unsupported TAMS database format: {self._format} !"
                _logger.error(err_msg)
                raise ValueError(err_msg)
            self._name = f"{self._path}"
            self._abs_path: Path = Path.cwd() / self._name
            self._sql_name = f"{self._name}/trajPool.db"
        else:
            self._sql_name = f".sqldb_tams_{np.random.default_rng().integers(0, 999999):06d}.db"

        self._store_archive = params.get("database", {}).get("archive_discarded", True)

        # Trajectory ensembles: one for active trajectories and one for
        # archived (discarded) members.
        # In-memory container
        self._trajs_db: list[Trajectory] = []
        self._archived_trajs_db: list[Trajectory] = []

        # Algorithm data
        self._init_ensemble_done = False

        # Initialize only metadata at this point
        # so that the object remains lightweight
        self._ntraj: int = ntraj
        self._nsplititer: int = nsplititer
        self._init_metadata()

    @classmethod
    def load(cls, a_path: Path, read_only: bool = True) -> Database:
        """Instantiate a TAMS database from disk.

        Args:
            a_path: the path to the database
            read_only: the database access mode

        Return:
            a TAMS database object
        """
        if not a_path.exists():
            err_msg = f"Database {a_path} does not exist !"
            _logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        # Load necessary elements to call the constructor
        db_params = toml.load(a_path / "input_params.toml")

        # If the a_path differs from the one stored in the
        # database (the DB has been moved), update the path
        if a_path.absolute().as_posix() != Path(db_params["database"]["path"]).absolute().as_posix():
            warn_msg = f"Database {db_params['database']['path']} has been moved to {a_path} !"
            _logger.warning(warn_msg)
            db_params["database"]["path"] = str(a_path)

        # Load picked forward model
        model_file = Path(a_path / "fmodel.pkl")
        with model_file.open("rb") as f:
            model = cloudpickle.load(f)

        return cls(model, db_params, read_only=read_only)

    def _init_metadata(self) -> None:
        """Initialize the database.

        Initialize database internal metadata (only) and setup
        the database on disk if needed.
        """
        # Initialize or load disk-based database metadata
        if self._save_to_disk:
            # Check for an existing database:
            db_exists = self._abs_path.exists()

            # If no previous db or we force restart
            # Overwrite the default read-only mode
            if not db_exists or self._restart:
                # The 'restart' is no longer useful, drop it
                self._parameters["database"].pop("restart", None)
                self._restart = False
                self._read_only = False
                self._setup_tree()

            # Load the database
            else:
                self._load_metadata()

                # Parameters stored in the DB override
                # newly provided parameters.
                with Path(self._abs_path / "input_params.toml").open("r") as f:
                    stored_params = toml.load(f)

                # Update input parameters that can be updated
                if self._parameters != stored_params:
                    self._update_run_params(stored_params)

            # Initialize the SQL pool file
            if self._read_only:
                self._pool_db = SQLFile(self.pool_file(), ro_mode=True)
            else:
                self._pool_db = SQLFile(self.pool_file())

        # Initialize in-memory database metadata
        # Overwrite default read-only mode
        else:
            self._read_only = False
            self._pool_db = SQLFile(self.pool_file())

        # Check minimal parameters
        if self._ntraj == -1 or self._nsplititer == -1:
            err_msg = "Initializing TAMS database missing ntraj and/or nsplititer parameter !"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def _update_run_params(self, old_params: dict[Any, Any]) -> None:
        """Update database params and metadata.

        Upon loading a database from disk, compare the dictionary of
        parameters stored in the database against the newly inputted one
        and update the database metadata when possible.
        Note that only the [tams] sub-dictionary can be updated updated at this point
        and the database params overwrite the other subdicts.

        Args:
            old_params: a dictionary of input parameter loaded from disk
        """
        # For testing purposes the params might be lacking
        # a "tams" subdir
        if "tams" not in old_params or "tams" not in self._parameters:
            # Simply overwrite the provided input params
            self._parameters.update(old_params)
            return

        # Update the number of splitting iteration
        self._nsplititer = self._parameters.get("tams", {}).get("nsplititer")
        old_params["tams"].update({"nsplititer": self._nsplititer})

        # If the initial ensemble of trajectory is not done
        # or we stopped after the initial ensemble stage
        if not self._init_ensemble_done or (
            self._init_ensemble_done and old_params["tams"].get("init_ensemble_only", False)
        ):
            self._ntraj = self._parameters.get("tams", {}).get("ntrajectories")
            old_params["tams"].update({"ntrajectories": self._ntraj})
            self._init_ensemble_done = False

        # Update other parameters in the [tams] subdir,
        # even if they do not change the database behavior
        for key, value in self._parameters["tams"].items():
            if key not in ["nsplititer", "ntrajectories"]:
                old_params["tams"][key] = value

        # Updated disk parameters overwrite the input params
        self._parameters.update(old_params)

        # Update the content of the database
        # if permitted
        if not self._read_only:
            self._write_metadata()
            with Path(self._abs_path / "input_params.toml").open("w") as f:
                toml.dump(self._parameters, f)

    def _setup_tree(self) -> None:
        """Initialize the trajectory database tree."""
        if self._save_to_disk:
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

            Path(self._name).mkdir()

            # Save the runtime options
            with Path(self._abs_path / "input_params.toml").open("w") as f:
                toml.dump(self._parameters, f)

            # Header file with metadata
            self._write_metadata()

            # Serialize the model
            # We need to pickle by value the local modules
            # which might not be available if we move the database
            # Note: only one import depth is handled at this point, we might
            #       want to make this recursive in the future
            model_file = Path(self._abs_path / "fmodel.pkl")
            cloudpickle.register_pickle_by_value(sys.modules[self._fmodel_t.__module__])
            for mods in get_module_local_import(self._fmodel_t.__module__):
                cloudpickle.register_pickle_by_value(sys.modules[mods])
            with model_file.open("wb") as f:
                cloudpickle.dump(self._fmodel_t, f)

            # Empty trajectories subfolder
            Path(self._abs_path / "trajectories").mkdir(parents=True)

    def _write_metadata(self) -> None:
        """Write the database Metadata to disk."""
        if self._format == "XML":
            header_file = self.header_file()
            root = ET.Element("header")
            mdata = ET.SubElement(root, "metadata")
            mdata.append(new_element("pyTAMS_version", version(__package__)))
            mdata.append(new_element("date", self._creation_date))
            mdata.append(new_element("model_t", self._fmodel_t.name()))
            mdata.append(new_element("ntraj", self._ntraj))
            mdata.append(new_element("nsplititer", self._nsplititer))
            mdata.append(new_element("init_ensemble_done", self._init_ensemble_done))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="\t", level=0)
            tree.write(header_file)
        else:
            err_msg = f"Unsupported TAMS database format: {self._format} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def _load_metadata(self) -> None:
        """Read the database Metadata from the header."""
        if self._save_to_disk:
            if self._format == "XML":
                tree = ET.parse(self.header_file())
                root = tree.getroot()
                mdata = root.find("metadata")
                datafromxml = xml_to_dict(mdata)
                self._ntraj = datafromxml["ntraj"]
                self._nsplititer = datafromxml["nsplititer"]
                self._init_ensemble_done = datafromxml["init_ensemble_done"]
                self._version = datafromxml["pyTAMS_version"]
                if self._version != version(__package__):
                    warn_msg = f"Database pyTAMS version {self._version} is different from {version(__package__)}"
                    _logger.warning(warn_msg)
                self._creation_date = datafromxml["date"]
                db_model = datafromxml["model_t"]
                if db_model != self._fmodel_t.name():
                    err_msg = f"Database model {db_model} is different from call {self._fmodel_t.name()}"
                    _logger.error(err_msg)
                    raise RuntimeError(err_msg)
            else:
                err_msg = f"Unsupported TAMS database format: {self._format} !"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def init_active_ensemble(self) -> None:
        """Initialize the requested number of trajectories."""
        for n in range(self._ntraj):
            workdir = Path(self._abs_path / f"trajectories/{form_trajectory_id(n)}") if self._save_to_disk else None
            t = Trajectory(
                traj_id=n,
                weight=1.0 / float(self._ntraj),
                fmodel_t=self._fmodel_t,
                parameters=self._parameters,
                workdir=workdir,
            )
            self.append_traj(t, True)

    def save_trajectory(self, traj: Trajectory) -> None:
        """Save a trajectory to disk in the database.

        Args:
            traj: the trajectory to save
        """
        if not self._save_to_disk:
            return

        traj.store()

    def load_data(self, load_archived_trajectories: bool = False) -> None:
        """Load data stored into the database.

        The initialization of the database only populate the metadata
        but not the full trajectories data.

        Args:
            load_archived_trajectories: whether to load archived trajectories
        """
        if not self._save_to_disk:
            return

        if not self._pool_db:
            err_msg = "Database is not initialized !"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        # Counter for number of trajectory loaded
        n_traj_restored = 0

        load_frozen = self._read_only

        ntraj_in_db = self._pool_db.get_trajectory_count()
        for n in range(ntraj_in_db):
            checkpath, metadata_str = self._pool_db.fetch_trajectory(n)
            traj_checkfile = Path(self._abs_path) / checkpath
            workdir = Path(self._abs_path / f"trajectories/{traj_checkfile.stem}")
            if traj_checkfile.exists():
                n_traj_restored += 1
                self.append_traj(
                    Trajectory.restore_from_checkfile(
                        traj_checkfile,
                        metadata_str,
                        fmodel_t=self._fmodel_t,
                        parameters=self._parameters,
                        workdir=workdir,
                        frozen=load_frozen,
                    ),
                    False,
                )
            else:
                t = Trajectory.init_from_metadata(
                    metadata_str,
                    fmodel_t=self._fmodel_t,
                    parameters=self._parameters,
                    workdir=workdir,
                )
                self.append_traj(t, False)

        if n_traj_restored > 0:
            inf_msg = f"{n_traj_restored} active trajectories loaded"
            _logger.info(inf_msg)

        # Load the archived trajectories if requested.
        # Those are loaded as 'frozen', i.e. the internal model
        # is not available and advance function disabled.
        if load_archived_trajectories:
            self.load_archived_trajectories()

        self.info()

    def load_archived_trajectories(self) -> None:
        """Load the archived trajectories data."""
        if not self._save_to_disk:
            return

        if not self._pool_db:
            err_msg = "Database is not initialized !"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        n_traj_restored = 0

        archived_ntraj_in_db = self._pool_db.get_archived_trajectory_count()
        for n in range(archived_ntraj_in_db):
            checkpath, metadata_str = self._pool_db.fetch_archived_trajectory(n)
            traj_checkfile = Path(self._abs_path) / checkpath
            workdir = Path(self._abs_path / f"trajectories/{traj_checkfile.stem}")
            if traj_checkfile.exists():
                n_traj_restored += 1
                self.append_archived_traj(
                    Trajectory.restore_from_checkfile(
                        traj_checkfile,
                        metadata_str,
                        fmodel_t=self._fmodel_t,
                        parameters=self._parameters,
                        workdir=workdir,
                        frozen=True,
                    ),
                    False,
                )

        inf_msg = f"{n_traj_restored} archived trajectories loaded"
        _logger.info(inf_msg)

    def name(self) -> str:
        """Accessor to DB name.

        Return:
            DB name
        """
        return self._name

    def append_traj(self, a_traj: Trajectory, update_db: bool) -> None:
        """Append a Trajectory to the internal list.

        Args:
            a_traj: the trajectory
            update_db: True to update the SQL DB content
        """
        # Also adds it to the SQL pool file.
        # and set the checkfile
        if self._save_to_disk:
            checkfile_str = f"./trajectories/{a_traj.idstr()}.xml"
            checkfile = Path(self._abs_path) / checkfile_str
            a_traj.set_checkfile(checkfile)
        else:
            checkfile_str = f"{a_traj.idstr()}.xml"
        if update_db:
            self._pool_db.add_trajectory(checkfile_str, a_traj.serialize_metadata_json())

        self._trajs_db.append(a_traj)

    def append_archived_traj(self, a_traj: Trajectory, update_db: bool) -> None:
        """Append an archived Trajectory to the internal list.

        Args:
            a_traj: the trajectory
            update_db: True to update the SQL DB content
        """
        checkfile_str = f"./trajectories/{a_traj.idstr()}.xml"
        checkfile = Path(self._abs_path) / checkfile_str
        a_traj.set_checkfile(checkfile)
        if update_db:
            self._pool_db.archive_trajectory(checkfile_str, a_traj.serialize_metadata_json())

        self._archived_trajs_db.append(a_traj)

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
        if idx < 0 or idx >= len(self._trajs_db):
            err_msg = f"Trying to access a non existing trajectory {idx} !"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        return self._trajs_db[idx]

    def overwrite_traj(self, idx: int, traj: Trajectory) -> None:
        """Deep copy a trajectory into internal list.

        Args:
            idx: the index of the trajectory to override
            traj: the new trajectory

        Raises:
            ValueError if idx is out of range
        """
        if idx < 0 or idx >= len(self._trajs_db):
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
        return self._sql_name

    def get_pool_db(self) -> SQLFile:
        """Get the pool SQL database handle."""
        return self._pool_db

    def is_empty(self) -> bool:
        """Check if list of trajectories is empty.

        Return:
            True if the list of trajectories is empty
        """
        return self._pool_db.get_trajectory_count() == 0

    def traj_list_len(self) -> int:
        """Length of the trajectory list.

        Return:
            Trajectory list length
        """
        return len(self._trajs_db)

    def archived_traj_list_len(self) -> int:
        """Length of the archived trajectory list.

        Return:
            Trajectory list length
        """
        if not self._store_archive:
            return 0

        return len(self._archived_trajs_db)

    def update_traj_list(self, a_traj_list: list[Trajectory]) -> None:
        """Overwrite the trajectory list.

        Args:
            a_traj_list: the new trajectory list
        """
        self._trajs_db = a_traj_list

    def archive_trajectory(self, traj: Trajectory) -> None:
        """Archive a trajectory about to be discarded.

        Args:
            traj: the trajectory to archive
        """
        if not self._store_archive:
            return

        # A branched trajectory will be overwritten by the
        # newly generated one in-place in the _trajs_db list.
        self._archived_trajs_db.append(traj)

        # Update the list of archived trajectories in the SQL DB
        checkfile_str = (
            traj.get_checkfile().relative_to(self._abs_path).as_posix()
            if self._save_to_disk
            else traj.get_checkfile().as_posix()
        )
        self._pool_db.archive_trajectory(checkfile_str, traj.serialize_metadata_json())

    def lock_trajectory(self, tid: int, allow_completed_lock: bool = False) -> bool:
        """Lock a trajectory in the SQL DB.

        Args:
            tid: the trajectory id
            allow_completed_lock: True if the trajectory can be locked even if it is completed

        Return:
            True if no disk DB and the trajectory was locked

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        return self._pool_db.lock_trajectory(tid, allow_completed_lock)

    def unlock_trajectory(self, tid: int, has_ended: bool) -> None:
        """Unlock a trajectory in the SQL DB.

        Args:
            tid: the trajectory id
            has_ended: True if the trajectory has ended

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        if has_ended:
            self._pool_db.mark_trajectory_as_completed(tid)
        else:
            self._pool_db.release_trajectory(tid)

    def update_trajectory(self, traj_id: int, traj: Trajectory) -> None:
        """Update a trajectory file in the DB.

        Args:
            traj_id : The trajectory id
            traj : the trajectory to get the data from

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        checkfile_str = traj.get_checkfile().relative_to(self._abs_path).as_posix()
        self._pool_db.update_trajectory(traj_id, checkfile_str, traj.serialize_metadata_json())

    def update_trajectories_weights(self) -> None:
        """Update the weights of all the trajectories.

        Using the the current splitting iteration weight.
        """
        tweight = self.weights()[-1] / self.n_traj()
        for t in self._trajs_db:
            t.set_weight(tweight)
            if self._save_to_disk:
                self._pool_db.update_trajectory_weight(t.id(), tweight)

    def weights(self) -> npt.NDArray[np.number]:
        """Splitting iterations weights."""
        return self._pool_db.get_weights()

    def append_splitting_iteration_data(
        self,
        ksplit: int,
        bias: int,
        discarded_ids: list[int],
        ancestor_ids: list[int],
        min_vals: list[float],
        min_max: list[float],
    ) -> None:
        """Append a set of splitting data to internal list.

        Args:
            ksplit : The splitting iteration index
            bias : The number of restarted trajectories, also ref. to as bias
            discarded_ids : The list of discarded trajectory ids
            ancestor_ids : The list of trajectories used to restart (ancestors)
            min_vals : The list of minimum values
            min_max : The score minimum and maximum values

        Raises:
            ValueError if the provided ksplit is incompatible with the db state
        """
        # Compute the weight of the ensemble at the current iteration
        # Insert 1.0 at the front of the weight array
        weights = np.insert(self._pool_db.get_weights(), 0, 1.0)
        new_weight = weights[-1] * (1.0 - bias / self._ntraj)

        # Check the splitting iteration index. If the incoming split is not
        # equal to the one in the database, something is wrong.
        if ksplit != self.k_split():
            self._pool_db.dump_file_json()
            err_msg = f"Attempting to add splitting iteration with splitting index {ksplit} \
                    incompatible with the last entry of the database {self.k_split()} !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        self._pool_db.add_splitting_data(ksplit, bias, new_weight, discarded_ids, ancestor_ids, min_vals, min_max)

    def update_splitting_iteration_data(
        self,
        ksplit: int,
        bias: int,
        discarded_ids: list[int],
        ancestor_ids: list[int],
        min_vals: list[float],
        min_max: list[float],
    ) -> None:
        """Update the last set of splitting data to internal list.

        Args:
            ksplit : The splitting iteration index
            bias : The number of restarted trajectories, also ref. to as bias
            discarded_ids : The list of discarded trajectory ids
            ancestor_ids : The list of trajectories used to restart (ancestors)
            min_vals : The list of minimum values
            min_max : The score minimum and maximum values

        Raises:
            ValueError if the provided ksplit is incompatible with the db state
        """
        # Compute the weight of the ensemble at the current iteration
        # Insert 1.0 at the front of the weight array
        weights = np.insert(self._pool_db.get_weights(), 0, 1.0)
        new_weight = weights[-1] * (1.0 - bias / self._ntraj)

        # Check the splitting iteration index. If the incoming split is not
        # equal to the one in the database, something is wrong.
        if (ksplit + bias) != self.k_split():
            self._pool_db.dump_file_json()
            err_msg = f"Attempting to update splitting iteration with splitting index {ksplit + bias} \
                    incompatible with the last entry of the database {self.k_split()} !"
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        self._pool_db.update_splitting_data(ksplit, bias, new_weight, discarded_ids, ancestor_ids, min_vals, min_max)

    def mark_last_splitting_iteration_as_done(self) -> None:
        """Flag the last splitting iteration as done."""
        self._pool_db.mark_last_iteration_as_completed()

    def n_traj(self) -> int:
        """Return the number of trajectory used for TAMS.

        Note that this is the requested number of trajectory, not
        the current length of the trajectory pool.

        Return:
            number of trajectory
        """
        return self._ntraj

    def n_split_iter(self) -> int:
        """Return the number of splitting iteration used for TAMS.

        Note that this is the requested number of splitting iteration, not
        the current splitting iteration.

        Return:
            number of splitting iteration
        """
        return self._nsplititer

    def path(self) -> str | None:
        """Return the path to the database."""
        if self._save_to_disk:
            return self._abs_path.absolute().as_posix()
        return None

    def done_with_splitting(self) -> bool:
        """Check if we are done with splitting."""
        return self.k_split() >= self._nsplititer

    def get_ongoing(self) -> list[int] | None:
        """Return the list of trajectories undergoing branching or None.

        Ongoing trajectories are extracted from the last splitting
        iteration data if it has not been flagged as "completed".
        """
        return self._pool_db.get_ongoing()

    def k_split(self) -> int:
        """Get the current splitting iteration index.

        The current splitting iteration index is equal to the
        ksplit + bias (number of branching event in the last iteration)
        entries of last entry in the SQL db table

        Returns:
            Internal splitting iteration index
        """
        return self._pool_db.get_k_split()

    def set_init_ensemble_flag(self, status: bool) -> None:
        """Change the initial ensemble status flag.

        Args:
            status: the new status
        """
        self._init_ensemble_done = status

        if self._save_to_disk:
            # Update the metadata file
            self._write_metadata()

    def init_ensemble_done(self) -> bool:
        """Get the initial ensemble status flag.

        Returns:
            the flag indicating that the initial ensemble is finished
        """
        return self._init_ensemble_done

    def count_ended_traj(self) -> int:
        """Return the number of trajectories that ended."""
        count = 0
        for i in range(self._pool_db.get_trajectory_count()):
            _, metadata = self._pool_db.fetch_trajectory(i)
            if Trajectory.deserialize_metadata(metadata)["ended"]:
                count += 1
        return count

    def count_converged_traj(self) -> int:
        """Return the number of trajectories that converged."""
        count = 0
        for i in range(self._pool_db.get_trajectory_count()):
            _, metadata = self._pool_db.fetch_trajectory(i)
            if Trajectory.deserialize_metadata(metadata)["converged"]:
                count += 1
        return count

    def count_computed_steps(self) -> int:
        """Return the total number of steps taken.

        This total count includes both the active and
        discarded trajectories.
        """
        count = 0
        for i in range(self._pool_db.get_trajectory_count()):
            _, metadata = self._pool_db.fetch_trajectory(i)
            count = count + Trajectory.deserialize_metadata(metadata)["nstep_compute"]

        for i in range(self._pool_db.get_archived_trajectory_count()):
            _, metadata = self._pool_db.fetch_archived_trajectory(i)
            count = count + Trajectory.deserialize_metadata(metadata)["nstep_compute"]

        return count

    def get_transition_probability(self) -> float:
        """Return the transition probability."""
        if self.count_ended_traj() < self._ntraj:
            return 0.0

        # Insert a first element to the weight array
        weights = np.insert(self._pool_db.get_weights(), 0, 1.0)
        biases = self._pool_db.get_biases()

        w = self._ntraj * weights[-1]
        for i in range(biases.shape[0]):
            w += biases[i] * weights[i]

        return float(self.count_converged_traj() * weights[-1] / w)

    def info(self) -> None:
        """Print database info to screen."""
        db_date_str = str(self._creation_date)
        pretty_line = "####################################################"
        inf_tbl = f"""
            {pretty_line}
            # TAMS v{self._version:17s} trajectory database      #
            # Date: {db_date_str:42s} #
            # Model: {self._fmodel_t.name():41s} #
            {pretty_line}
            # Requested # of traj: {self._ntraj:27} #
            # Requested # of splitting iter: {self._nsplititer:17} #
            # Number of 'Ended' trajectories: {self.count_ended_traj():16} #
            # Number of 'Converged' trajectories: {self.count_converged_traj():12} #
            # Current splitting iter counter: {self.k_split():16} #
            # Current total number of steps: {self.count_computed_steps():17} #
            # Transition probability: {self.get_transition_probability():24} #
            {pretty_line}
        """
        _logger.info(inf_tbl)

    def reset_initial_ensemble_stage(self) -> None:
        """Reset the database content to the initial ensemble stage.

        In particular, the splitting iteration data is cleared, the
        list of active trajectories restored and any branched trajectory
        data deleted.

        First, the active list in the SQL db is updated, the archived list
        cleared and a new call to load_data update the in-memory data.
        """
        if self.traj_list_len() == 0 or not self._pool_db:
            err_msg = "Database data not loaded or empty. Try load_data()"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        # Update the active trajectories list in the SQL db
        # while deleting the checkfiles from the trajectory we updaate
        for t in self._trajs_db:
            tid = t.id()
            if t.get_nbranching() > 0:
                for at in self._archived_trajs_db:
                    if at.id() == tid and at.get_nbranching() == 0:
                        self.update_trajectory(tid, at)
                        t.delete()

        # Delete checkfiles of all the archived trajectories we did not
        # restored as active
        for at in self._archived_trajs_db:
            if at.get_nbranching() > 0:
                at.delete()

        # Clear the obsolete content of the SQL file
        self._pool_db.clear_archived_trajectories()
        self._pool_db.clear_splitting_data()

        self._trajs_db.clear()
        self._archived_trajs_db.clear()

        # Reload the data
        self.load_data()

    def plot_score_functions(self, fname: str | None = None, plot_archived: bool = False) -> None:
        """Plot the score as function of time for all trajectories."""
        pltfile = fname if fname else Path(self._name).stem + "_scores.png"

        plt.figure(figsize=(10, 6))
        for t in self._trajs_db:
            plt.plot(t.get_time_array(), t.get_score_array(), linewidth=0.8)

        if plot_archived:
            for t in self._archived_trajs_db:
                plt.plot(t.get_time_array(), t.get_score_array(), linewidth=0.8)

        plt.xlabel(r"$Time$", fontsize="x-large")
        plt.xlim(left=0.0)
        plt.ylabel(r"$Score \; [-]$", fontsize="x-large")
        plt.xticks(fontsize="x-large")
        plt.yticks(fontsize="x-large")
        plt.grid(linestyle="dotted")
        plt.tight_layout()  # to fit everything in the prescribed area
        plt.savefig(pltfile, dpi=300)
        plt.clf()
        plt.close()

    def plot_min_max_span(self, fname: str | None = None) -> None:
        """Plot the evolution of the ensemble min/max during iterations."""
        pltfile = fname if fname else Path(self._name).stem + "_minmax.png"

        plt.figure(figsize=(6, 4))

        min_max_data = self._pool_db.get_minmax()
        plt.plot(min_max_data[:, 0], min_max_data[:, 1], linewidth=1.0, label="min of maxes")
        plt.plot(min_max_data[:, 0], min_max_data[:, 2], linewidth=1.0, label="max of maxes")
        plt.grid(linestyle="dotted")
        ax = plt.gca()
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, np.max(min_max_data[:, 0]))
        ax.legend()
        plt.tight_layout()
        plt.savefig(pltfile, dpi=300)
        plt.clf()
        plt.close()

    def _get_location_and_indices_at_k(self, k_in: int) -> list[tuple[str, int]]:
        """Return the location and indices of active trajectory at k_in.

        Location here can be either 'active' or "archive" depending on
        whether the trajectory we are interested in is still in the current
        active list or in the archived list.

        Args:
            k_in : the index of the splitting iteration

        Returns:
            A list of tuple with the location and index of each trajectory
            active at iteration k
        """
        # Initialize active @k list with current active list
        # For now handle tuple with (active/archived, index)
        # The actual trajectory list will be filled later
        active_list_index = [("active", i) for i in range(self._ntraj)]

        # Traverse in reverse the splitting iteration table
        idx_in_archive = self.archived_traj_list_len() - 1
        for k in range(self._pool_db.get_iteration_count() - 1, k_in - 1, -1):
            splitting_data = self._pool_db.fetch_splitting_data(k)
            if splitting_data:
                _, nbranch, _, discarded, _, _, _, status = splitting_data
                if status == "locked":
                    continue
                for discarded_idx in discarded:
                    for i in range(idx_in_archive, idx_in_archive - nbranch, -1):
                        if self._archived_trajs_db[i].id() == discarded_idx:
                            active_list_index[discarded_idx] = ("archive", i)
            idx_in_archive = idx_in_archive - nbranch

        return active_list_index

    def get_trajectory_active_at_k(self, k_in: int) -> list[Trajectory]:
        """Return the list of trajectory active at a given splitting iteration.

        To explore the ensemble evolution during splitting iterations, it is
        useful to reconstruct the list of active trajectories at the beginning
        of any given splitting iteration.

        Note that k here is not the splitting index, but the iteration index.
        Since more than one child can be spawned at each splitting iteration,
        the two might differ.

        Args:
            k_in : the index of the splitting iteration

        Returns:
            The list of trajectories active at the beginning of iteration k
        """
        # Check that the requested index is available in the database
        if k_in >= self._pool_db.get_iteration_count():
            err_msg = (
                f"Attempting to read splitting iteration {k_in} data"
                f"larger than stored data {self._pool_db.get_iteration_count()}"
            )
            _logger.exception(err_msg)
            raise ValueError(err_msg)

        # Check that archived trajectories are stored
        if self.archived_traj_list_len() == 0:
            err_msg = "Cannot reconstruct active set without stored archives !"
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        # First get the location and indices of the trajectories
        # active at iteration k
        active_list_index = self._get_location_and_indices_at_k(k_in)

        # Retrieve the trajectories from the active/archived lists
        active_list_at_k = []
        for location, idx in active_list_index:
            if location == "active":
                active_list_at_k.append(self._trajs_db[idx])
            elif location == "archive":
                active_list_at_k.append(self._archived_trajs_db[idx])

        return active_list_at_k

    def __del__(self) -> None:
        """Destructor of the db.

        Delete the hidden SQL database if we do not intend to keep
        the database around.
        """
        # Even if we plan to keep the SQL database around, force
        # deleting the SQL connection
        if hasattr(self, "_pool_db"):
            del self._pool_db
            # Remove the hidden db file
            if not self._save_to_disk:
                Path(self.pool_file()).unlink(missing_ok=True)
