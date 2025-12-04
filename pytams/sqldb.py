"""A class for the TAMS data as an SQL database using SQLAlchemy."""

from __future__ import annotations
import gc
import json
import logging
from pathlib import Path
from typing import cast
import numpy as np
import numpy.typing as npt
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import sessionmaker

_logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """A base class for the tables."""


class Trajectory(Base):
    """A table storing the active trajectories."""

    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    traj_file: Mapped[str] = mapped_column(nullable=False)
    t_metadata: Mapped[str] = mapped_column(default="", nullable=False)
    status: Mapped[str] = mapped_column(default="idle", nullable=False)


class ArchivedTrajectory(Base):
    """A table storing the archived trajectories."""

    __tablename__ = "archived_trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    traj_file: Mapped[str] = mapped_column(nullable=False)
    t_metadata: Mapped[str] = mapped_column(default="", nullable=False)


class SplittingIterations(Base):
    """A table storing the splitting iterations."""

    __tablename__ = "splitting_iterations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    split_id: Mapped[int] = mapped_column(nullable=False)
    bias: Mapped[int] = mapped_column(nullable=False)
    weight: Mapped[str] = mapped_column(nullable=False)
    discarded_traj_ids: Mapped[str] = mapped_column(nullable=False)
    ancestor_traj_ids: Mapped[str] = mapped_column(nullable=False)
    min_vals: Mapped[str] = mapped_column(nullable=False)
    min_max: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(default="locked", nullable=False)


valid_statuses = ["locked", "idle", "completed"]


class SQLFile:
    """An SQL file.

    Allows atomic access to an SQL database from all
    the workers.

    Note: TAMS works with Python indexing starting at 0,
    while SQL indexing starts at 1. Trajectory ID is
    updated accordingly when accessing/updating the DB.

    Attributes:
        _file_name : The file name
    """

    def __init__(self, file_name: str, in_memory: bool = False, ro_mode: bool = False) -> None:
        """Initialize the file.

        Args:
            file_name : The file name
            in_memory: a bool to trigger in-memory creation
            ro_mode: a bool to trigger read-only access to the database
        """
        self._file_name = "" if in_memory else file_name

        # URI mode requires absolute path
        file_path = Path(file_name).absolute().as_posix()
        if in_memory:
            self._engine = create_engine("sqlite:///:memory:", echo=False)
        else:
            self._engine = (
                create_engine(f"sqlite:///file:{file_path}?mode=ro&uri=true", echo=False)
                if ro_mode
                else create_engine(f"sqlite:///{file_path}", echo=False)
            )
        self._Session = sessionmaker(bind=self._engine)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the tables of the file.

        Raises:
            RuntimeError : If a connection to the DB could not be acquired
        """
        try:
            Base.metadata.create_all(self._engine)
        except SQLAlchemyError:
            err_msg = "Failed to initialize DB schema"
            _logger.exception(err_msg)
            raise

    def name(self) -> str:
        """Access the DB file name.

        Returns:
            the database name, empty string if in-memory
        """
        return self._file_name

    def add_trajectory(self, traj_file: str, metadata: str) -> None:
        """Add a new trajectory to the DB.

        Args:
            traj_file : The trajectory file of that trajectory
            metadata: a json representation of the traj metadata

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        session = self._Session()
        try:
            new_traj = Trajectory(traj_file=traj_file, t_metadata=metadata)
            session.add(new_traj)
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to add trajectory")
            raise
        finally:
            session.close()

    def update_trajectory(self, traj_id: int, traj_file: str, metadata: str) -> None:
        """Update a given trajectory data in the DB.

        Args:
            traj_id : The trajectory id
            traj_file : The new trajectory file of that trajectory
            metadata: a json representation of the traj metadata

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = traj_id + 1
            traj = session.query(Trajectory).filter(Trajectory.id == db_id).one()
            traj.traj_file = mapped_column(traj_file)
            traj.t_metadata = metadata
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            err_msg = f"Failed to update trajectory {traj_id}"
            _logger.exception(err_msg)
            raise
        finally:
            session.close()

    def update_trajectory_weight(self, traj_id: int, weight: float) -> None:
        """Update a given trajectory weight in the DB.

        Args:
            traj_id : The trajectory id
            weight: the new trajectory weight

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = traj_id + 1
            traj = session.query(Trajectory).filter(Trajectory.id == db_id).one()
            metadata_d = json.loads(traj.t_metadata)
            metadata_d["weight"] = str(weight)
            traj.t_metadata = mapped_column(json.dumps(metadata_d))
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            err_msg = f"Failed to update trajectory {traj_id} weight"
            _logger.exception(err_msg)
            raise
        finally:
            session.close()

    def lock_trajectory(self, traj_id: int, allow_completed_lock: bool = False) -> bool:
        """Set the status of a trajectory to "locked" if possible.

        Args:
            traj_id : The trajectory id
            allow_completed_lock : Allow to lock a "completed" trajectory

        Return:
            True if the trajectory was successfully locked, False otherwise

        Raises:
            ValueError if the trajectory with the given id does not exist
            SQLAlchemyError if the DB could not be accessed
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = traj_id + 1
            traj = session.query(Trajectory).filter(Trajectory.id == db_id).with_for_update().one_or_none()

            if traj:
                allowed_status = ["idle", "completed"] if allow_completed_lock else ["idle"]
                if traj.status in allowed_status:
                    traj.status = "locked"
                    session.commit()
                    return True
                return False

            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to lock trajectory")
            raise
        finally:
            session.close()

    def mark_trajectory_as_completed(self, traj_id: int) -> None:
        """Set the status of a trajectory to "completed" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
            SQLAlchemyError if the DB could not be accessed
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = traj_id + 1
            traj = session.query(Trajectory).filter(Trajectory.id == db_id).one_or_none()
            if traj:
                if traj.status in ["locked"]:
                    traj.status = "completed"
                    session.commit()
                else:
                    warn_msg = f"Attempting to mark completed Trajectory {traj_id} already in status {traj.status}."
                    _logger.warning(warn_msg)
            else:
                err_msg = f"Trajectory {traj_id} does not exist"
                _logger.error(err_msg)
                raise ValueError(err_msg)
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to mark trajectory as completed")
            raise
        finally:
            session.close()

    def release_trajectory(self, traj_id: int) -> None:
        """Set the status of a trajectory to "idle" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = traj_id + 1
            traj = session.query(Trajectory).filter(Trajectory.id == db_id).one_or_none()
            if traj:
                if traj.status in ["locked"]:
                    traj.status = "idle"
                    session.commit()
                else:
                    warn_msg = f"Attempting to release Trajectory {traj_id} already in status {traj.status}."
                    _logger.warning(warn_msg)
            else:
                err_msg = f"Trajectory {traj_id} does not exist"
                _logger.error(err_msg)
                raise ValueError(err_msg)
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to release trajectory")
            raise
        finally:
            session.close()

    def get_trajectory_count(self) -> int:
        """Get the number of trajectories in the DB.

        Returns:
            The number of trajectories
        """
        session = self._Session()
        try:
            return session.query(Trajectory).count()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to count the number of trajectories")
            raise
        finally:
            session.close()

    def fetch_trajectory(self, traj_id: int) -> tuple[str, str]:
        """Get the trajectory file of a trajectory.

        Args:
            traj_id : The trajectory id

        Return:
            The trajectory file

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = traj_id + 1
            traj = session.query(Trajectory).filter(Trajectory.id == db_id).one_or_none()
            if traj:
                tfile: str = traj.traj_file
                metadata_str: str = traj.t_metadata
                return tfile, metadata_str

            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to fetch trajectory")
            raise
        finally:
            session.close()

    def release_all_trajectories(self) -> None:
        """Release all trajectories in the DB."""
        session = self._Session()
        try:
            session.query(Trajectory).update({"status": "idle"})
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to release all trajectories")
        finally:
            session.close()

    def archive_trajectory(self, traj_file: str, metadata: str) -> None:
        """Add a new trajectory to the archive container.

        Args:
            traj_file : The trajectory file of that trajectory
            metadata: a json representation of the traj metadata
        """
        session = self._Session()
        try:
            new_traj = ArchivedTrajectory(traj_file=traj_file, t_metadata=metadata)
            session.add(new_traj)
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to archive trajectory")
        finally:
            session.close()

    def fetch_archived_trajectory(self, traj_id: int) -> tuple[str, str]:
        """Get the trajectory file of a trajectory in the archive.

        Args:
            traj_id : The trajectory id

        Return:
            The trajectory file

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = traj_id + 1
            traj = session.query(ArchivedTrajectory).filter(ArchivedTrajectory.id == db_id).one_or_none()
            if traj:
                tfile: str = traj.traj_file
                metadata_str: str = traj.t_metadata
                return tfile, metadata_str

            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to fetch archived trajectory")
            raise
        finally:
            session.close()

    def get_archived_trajectory_count(self) -> int:
        """Get the number of trajectories in the archive.

        Returns:
            The number of trajectories
        """
        session = self._Session()
        try:
            return session.query(ArchivedTrajectory).count()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to count the number of archived trajectories")
            raise
        finally:
            session.close()

    def clear_archived_trajectories(self) -> int:
        """Delete the content of the archived traj table.

        Returns:
            The number of entries deleted
        """
        session = self._Session()
        try:
            ndelete = session.query(ArchivedTrajectory).delete()
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to clear archived trajectories")
            raise
        else:
            return ndelete
        finally:
            session.close()

    def add_splitting_data(
        self,
        k: int,
        bias: int,
        weight: float,
        discarded_ids: list[int],
        ancestor_ids: list[int],
        min_vals: list[float],
        min_max: list[float],
    ) -> None:
        """Add a new splitting data to the DB.

        Args:
            k : The splitting iteration index
            bias : The number of restarted trajectories
            weight : Weight of the ensemble at the current iteration
            discarded_ids : The list of discarded trajectory ids
            ancestor_ids : The list of trajectories used to restart
            min_vals : The list of minimum values
            min_max : The score minimum and maximum values
        """
        session = self._Session()
        try:
            new_split = SplittingIterations(
                split_id=k,
                bias=bias,
                weight=str(weight),
                discarded_traj_ids=json.dumps(discarded_ids),
                ancestor_traj_ids=json.dumps(ancestor_ids),
                min_vals=json.dumps(min_vals),
                min_max=json.dumps(min_max),
            )
            session.add(new_split)
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to add splitting data")
            raise
        finally:
            session.close()

    def update_splitting_data(
        self,
        k: int,
        bias: int,
        weight: float,
        discarded_ids: list[int],
        ancestor_ids: list[int],
        min_vals: list[float],
        min_max: list[float],
    ) -> None:
        """Update the last splitting data row to the DB.

        Args:
            k : The splitting iteration index
            bias : The number of restarted trajectories
            weight : Weight of the ensemble at the current iteration
            discarded_ids : The list of discarded trajectory ids
            ancestor_ids : The list of trajectories used to restart
            min_vals : The list of minimum values
            min_max : The score minimum and maximum values
        """
        session = self._Session()
        try:
            dset = session.query(SplittingIterations).order_by(SplittingIterations.id.desc()).first()
            if dset:
                dset.split_id = k
                dset.bias = bias
                dset.weight = str(weight)
                dset.discarded_traj_ids = json.dumps(discarded_ids)
                dset.ancestor_traj_ids = json.dumps(ancestor_ids)
                dset.min_vals = json.dumps(min_vals)
                dset.min_max = json.dumps(min_max)
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to update the last splitting data")
            raise
        finally:
            session.close()

    def mark_last_iteration_as_completed(self) -> None:
        """Mark the last splitting iteration as complete.

        By default, iteration data append to the SQL table with a state "locked"
        to indicate an iteration being worked on. Upon completion, mark it as
        "completed" otherwise the iteration is considered incomplete, i.e.
        interrupted by some error or wall clock limit.
        """
        session = self._Session()
        try:
            iteration = session.query(SplittingIterations).order_by(SplittingIterations.id.desc()).first()
            if iteration:
                iteration.status = "completed"
                session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to mark splitting iteration as completed")
            raise
        finally:
            session.close()

    def get_k_split(self) -> int:
        """Get the current splitting iteration counter.

        Returns:
            The ksplit from the last entry in the SplittingIterations table
        """
        session = self._Session()
        try:
            last_split = session.query(SplittingIterations).order_by(SplittingIterations.id.desc()).first()
            if last_split:
                return last_split.split_id + last_split.bias
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to query k_split !")
            raise
        else:
            return 0
        finally:
            session.close()

    def get_iteration_count(self) -> int:
        """Get the number of splitting iteration stored.

        Returns:
            The length of the SplittingIterations table
        """
        session = self._Session()
        try:
            return session.query(SplittingIterations).count()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to count the number of splitting iteration stored")
            raise
        finally:
            session.close()

    def fetch_splitting_data(
        self, k_id: int
    ) -> tuple[int, int, float, list[int], list[int], list[float], list[float], str] | None:
        """Get the splitting iteration data for a given iteration.

        Args:
            k_id : The iteration id

        Return:
            The splitting iteration data

        Raises:
            ValueError if the splitting iteration with the given id does not exist
        """
        session = self._Session()
        try:
            # SQL indexing starts at 1, adjust ID
            db_id = k_id + 1
            split = session.query(SplittingIterations).filter(SplittingIterations.id == db_id).one_or_none()
            if split:
                return (
                    int(split.split_id),
                    int(split.bias),
                    float(split.weight),
                    cast("list[int]", json.loads(split.discarded_traj_ids)),
                    cast("list[int]", json.loads(split.ancestor_traj_ids)),
                    cast("list[float]", json.loads(split.min_vals)),
                    cast("list[float]", json.loads(split.min_max)),
                    split.status,
                )

            err_msg = f"Splitting iteration {k_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to fetch splitting iteration data")
            raise
        finally:
            session.close()

    def get_ongoing(self) -> list[int] | None:
        """Get the list of ongoing trajectories if any.

        Returns:
            Either a list trajectories or None if nothing was left to do
        """
        session = self._Session()
        try:
            last_split = session.query(SplittingIterations).order_by(SplittingIterations.id.desc()).first()
            if last_split and last_split.status == "locked":
                return cast("list[int]", json.loads(last_split.discarded_traj_ids))
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to query the list of ongoing trajectories !")
            raise
        else:
            return None
        finally:
            session.close()

    def get_weights(self) -> npt.NDArray[np.number]:
        """Read the weights from the database.

        Returns:
            the weight for each splitting iteration as a numpy array
        """
        session = self._Session()
        try:
            return np.array(
                [r[0] for r in session.query(SplittingIterations).with_entities(SplittingIterations.weight).all()],
                dtype="float32",
            )
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to access the weights !")
            raise
        finally:
            session.close()

    def get_biases(self) -> npt.NDArray[np.number]:
        """Read the biases from the database.

        Returns:
            the bias for each splitting iteration as a numpy array
        """
        session = self._Session()
        try:
            return np.array(
                [r[0] for r in session.query(SplittingIterations).with_entities(SplittingIterations.bias).all()],
                dtype="int",
            )
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to access the biases !")
            raise
        finally:
            session.close()

    def get_minmax(self) -> npt.NDArray[np.number]:
        """Read the min/max from the database.

        Returns:
            the 2D Numpy array with k_index, min, max
        """
        session = self._Session()
        try:
            return np.array(
                [
                    [r[0], json.loads(r[1])[0], json.loads(r[1])[1]]
                    for r in session.query(SplittingIterations)
                    .with_entities(SplittingIterations.split_id, SplittingIterations.min_max)
                    .all()
                ],
                dtype="float32",
            )
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to access the biases !")
            raise
        finally:
            session.close()

    def clear_splitting_data(self) -> int:
        """Delete the content of the splitting data table.

        Returns:
            The number of entries deleted
        """
        session = self._Session()
        try:
            ndelete = session.query(SplittingIterations).delete()
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to clear splitting iterations data.")
            raise
        else:
            return ndelete
        finally:
            session.close()

    def dump_file_json(self, json_file: str | None = None) -> None:
        """Dump the content of the trajectory table to a json file.

        Args:
            json_file: an optional file name (or path) to dump the data to
        """
        db_data = {}
        session = self._Session()
        try:
            db_data["trajectories"] = {
                traj.id - 1: {"file": traj.traj_file, "status": traj.status, "metadata": traj.t_metadata}
                for traj in session.query(Trajectory).all()
            }
            db_data["archived_trajectories"] = {
                traj.id - 1: {"file": traj.traj_file, "metadata": traj.t_metadata}
                for traj in session.query(ArchivedTrajectory).all()
            }
            db_data["splitting_data"] = {
                split.id: {
                    "k": str(split.split_id),
                    "bias": str(split.bias),
                    "weight": split.weight,
                    "min_max_start": json.loads(split.min_max),
                    "discarded_ids": json.loads(split.discarded_traj_ids),
                    "ancestor_ids": json.loads(split.ancestor_traj_ids),
                    "min_vals": json.loads(split.min_vals),
                    "status": split.status,
                }
                for split in session.query(SplittingIterations).all()
            }
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to query the content of the DB")
            raise
        finally:
            session.close()

        json_path = Path(json_file) if json_file else Path(f"{Path(self._file_name).stem}.json")
        with json_path.open("w") as f:
            json.dump(db_data, f, indent=2)

    def __del__(self) -> None:
        """Explicit delete function.

        On windows, the SQL file is locked.
        """
        del self._Session
        self._engine.dispose()
        gc.collect()
