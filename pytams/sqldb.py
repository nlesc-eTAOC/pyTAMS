"""A class for the TAMS data as an SQL database using SQLAlchemy."""

from __future__ import annotations
import gc
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
import numpy.typing as npt
from sqlalchemy import JSON
from sqlalchemy import create_engine
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import Session
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

_logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """A base class for the tables."""


class Trajectory(Base):
    """A table storing the active trajectories."""

    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    traj_file: Mapped[str] = mapped_column(nullable=False)
    t_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(default="idle", nullable=False)


class ArchivedTrajectory(Base):
    """A table storing the archived trajectories."""

    __tablename__ = "archived_trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    traj_file: Mapped[str] = mapped_column(nullable=False)
    t_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class SplittingIterations(Base):
    """A table storing the splitting iterations."""

    __tablename__ = "splitting_iterations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    split_id: Mapped[int] = mapped_column(nullable=False)
    bias: Mapped[int] = mapped_column(nullable=False)
    weight: Mapped[float] = mapped_column(nullable=False)

    discarded_traj_ids: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    ancestor_traj_ids: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    min_vals: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    min_max: Mapped[list[float]] = mapped_column(JSON, nullable=False)
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
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
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

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def name(self) -> str:
        """Access the DB file name.

        Returns:
            the database name, empty string if in-memory
        """
        return self._file_name

    def add_trajectory(self, traj_file: str, metadata: dict) -> None:
        """Add a new trajectory to the DB.

        Args:
            traj_file : The trajectory file of that trajectory
            metadata: a dict with the metadata

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            new_traj = Trajectory(traj_file=traj_file, t_metadata=metadata)
            session.add(new_traj)

    def update_trajectory(self, traj_id: int, traj_file: str, metadata: dict) -> None:
        """Update a given trajectory data in the DB.

        Args:
            traj_id : The trajectory id
            traj_file : The new trajectory file of that trajectory
            metadata: a dict with the trajectory metadata

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            if traj:
                traj.traj_file = traj_file
                traj.t_metadata = metadata
            else:
                err_msg = f"Trajectory {traj_id} not found !"
                _logger.exception(err_msg)
                raise ValueError(err_msg)

    def update_trajectory_weight(self, traj_id: int, weight: float) -> None:
        """Update a given trajectory weight in the DB.

        Args:
            traj_id : The trajectory id
            weight: the new trajectory weight

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            if traj is None:
                err_msg = f"Trajectory {traj_id} not found !"
                _logger.exception(err_msg)
                raise ValueError(err_msg)

            metadata_d = dict(traj.t_metadata)
            metadata_d["weight"] = weight
            traj.t_metadata = metadata_d

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
        with self.session_scope() as session:
            stmt = select(Trajectory).filter(Trajectory.id == traj_id + 1).with_for_update()
            traj = session.execute(stmt).scalar_one_or_none()

            if traj:
                allowed_status = ["idle", "completed"] if allow_completed_lock else ["idle"]
                if traj.status in allowed_status:
                    traj.status = "locked"
                    return True
                return False

            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def mark_trajectory_as_completed(self, traj_id: int) -> None:
        """Set the status of a trajectory to "completed" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
            SQLAlchemyError if the DB could not be accessed
        """
        with self.session_scope() as session:
            traj = session.execute(select(Trajectory).filter(Trajectory.id == traj_id + 1)).scalar_one_or_none()
            if traj:
                if traj.status == "locked":
                    traj.status = "completed"
                else:
                    warn_msg = f"Attempting to mark completed Trajectory {traj_id} already in status {traj.status}."
                    _logger.warning(warn_msg)
            else:
                err_msg = f"Trajectory {traj_id} does not exist"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def release_trajectory(self, traj_id: int) -> None:
        """Set the status of a trajectory to "idle" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        with self.session_scope() as session:
            traj = session.execute(select(Trajectory).filter(Trajectory.id == traj_id + 1)).scalar_one_or_none()
            if traj:
                if traj.status == "locked":
                    traj.status = "idle"
                else:
                    warn_msg = f"Attempting to release Trajectory {traj_id} already in status {traj.status}."
                    _logger.warning(warn_msg)
            else:
                err_msg = f"Trajectory {traj_id} does not exist"
                _logger.error(err_msg)
                raise ValueError(err_msg)

    def get_trajectory_count(self) -> int:
        """Get the number of trajectories in the DB.

        Returns:
            The number of trajectories
        """
        with self.session_scope() as session:
            return session.scalar(select(func.count(Trajectory.id))) or 0

    def get_ended_trajectory_count(self) -> int:
        """Return the number of trajectories that have 'ended' in their metadata."""
        with self.session_scope() as session:
            stmt = select(func.count(Trajectory.id)).where(Trajectory.t_metadata["ended"].as_boolean())
            return session.scalar(stmt) or 0

    def get_converged_trajectory_count(self) -> int:
        """Return the number of trajectories that have 'converged' in their metadata."""
        with self.session_scope() as session:
            stmt = select(func.count(Trajectory.id)).where(Trajectory.t_metadata["converged"].as_boolean())
            return session.scalar(stmt) or 0

    def get_total_computed_steps(self) -> int:
        """Sum the 'nstep_compute' field across all active and archived trajectories."""
        with self.session_scope() as session:
            # Create a subquery for active trajectories
            active_steps = select(Trajectory.t_metadata["nstep_compute"].as_integer().label("steps"))

            # Create a subquery for archived trajectories
            archived_steps = select(ArchivedTrajectory.t_metadata["nstep_compute"].as_integer().label("steps"))

            # Combine them using union_all
            combined = active_steps.union_all(archived_steps).subquery()

            # Select the sum of the combined column
            total_sum = session.scalar(select(func.sum(combined.c.steps)))

            return int(total_sum) if total_sum else 0

    def fetch_trajectory(self, traj_id: int) -> tuple[str, dict]:
        """Get the trajectory file of a trajectory.

        Args:
            traj_id : The trajectory id

        Return:
            The trajectory file

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        with self.session_scope() as session:
            traj = session.get(Trajectory, traj_id + 1)
            if traj:
                return traj.traj_file, traj.t_metadata

            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def release_all_trajectories(self) -> None:
        """Release all trajectories in the DB."""
        with self.session_scope() as session:
            session.execute(update(Trajectory).values(status="idle"))

    def archive_trajectory(self, traj_file: str, metadata: dict) -> None:
        """Add a new trajectory to the archive container.

        Args:
            traj_file : The trajectory file of that trajectory
            metadata: a dict with the traj metadata
        """
        with self.session_scope() as session:
            new_traj = ArchivedTrajectory(traj_file=traj_file, t_metadata=metadata)
            session.add(new_traj)

    def fetch_archived_trajectory(self, traj_id: int) -> tuple[str, dict]:
        """Get the trajectory file of a trajectory in the archive.

        Args:
            traj_id : The trajectory id

        Return:
            The trajectory file

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        with self.session_scope() as session:
            db_id = traj_id + 1
            traj = session.get(ArchivedTrajectory, db_id)
            if traj:
                return traj.traj_file, traj.t_metadata

            err_msg = f"Archived Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def get_archived_trajectory_count(self) -> int:
        """Get the number of trajectories in the archive.

        Returns:
            The number of trajectories
        """
        with self.session_scope() as session:
            return session.scalar(select(func.count(ArchivedTrajectory.id))) or 0

    def clear_archived_trajectories(self) -> int:
        """Delete the content of the archived traj table.

        Returns:
            The number of entries deleted
        """
        with self.session_scope() as session:
            stmt = delete(ArchivedTrajectory)
            result = session.execute(stmt)
            return result.rowcount

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
        with self.session_scope() as session:
            new_split = SplittingIterations(
                split_id=k,
                bias=bias,
                weight=weight,
                discarded_traj_ids=discarded_ids,
                ancestor_traj_ids=ancestor_ids,
                min_vals=min_vals,
                min_max=min_max,
            )
            session.add(new_split)

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
        with self.session_scope() as session:
            stmt = select(SplittingIterations).order_by(SplittingIterations.id.desc())
            dset = session.execute(stmt).scalars().first()
            if dset:
                dset.split_id = k
                dset.bias = bias
                dset.weight = weight
                dset.discarded_traj_ids = discarded_ids
                dset.ancestor_traj_ids = ancestor_ids
                dset.min_vals = min_vals
                dset.min_max = min_max

    def mark_last_iteration_as_completed(self) -> None:
        """Mark the last splitting iteration as complete.

        By default, iteration data append to the SQL table with a state "locked"
        to indicate an iteration being worked on. Upon completion, mark it as
        "completed" otherwise the iteration is considered incomplete, i.e.
        interrupted by some error or wall clock limit.
        """
        with self.session_scope() as session:
            stmt = select(SplittingIterations).order_by(SplittingIterations.id.desc())
            iteration = session.execute(stmt).scalars().first()
            if iteration:
                iteration.status = "completed"

    def get_k_split(self) -> int:
        """Get the current splitting iteration counter.

        Returns:
            The ksplit from the last entry in the SplittingIterations table
        """
        with self.session_scope() as session:
            last_split = session.query(SplittingIterations).order_by(SplittingIterations.id.desc()).first()
            if last_split:
                return last_split.split_id + last_split.bias
            return 0

    def check_new_min_of_maxes(self, newmin: float) -> None:
        """Compare the incoming min to the last entry.

        When running TAMS, at each new iteration the ensemble minimum
        of maximum should be strictly above the previous iteration's one.

        Args:
            newmin: the new minimum of maximums
        """
        with self.session_scope() as session:
            last_split = session.query(SplittingIterations).order_by(SplittingIterations.id.desc()).first()
            if last_split:
                old_min = last_split.min_max[0]
                if newmin <= old_min:
                    wrn_msg = f"New iteration has minimum level {newmin} lower than old one {old_min}"
                    _logger.warning(wrn_msg)

    def get_iteration_count(self) -> int:
        """Get the number of splitting iteration stored.

        Returns:
            The length of the SplittingIterations table
        """
        with self.session_scope() as session:
            return session.scalar(select(func.count(SplittingIterations.id))) or 0

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
        with self.session_scope() as session:
            split = session.get(SplittingIterations, k_id + 1)
            if split:
                return (
                    split.split_id,
                    split.bias,
                    split.weight,
                    split.discarded_traj_ids,
                    split.ancestor_traj_ids,
                    split.min_vals,
                    split.min_max,
                    split.status,
                )

            err_msg = f"Splitting iteration {k_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def get_ongoing(self) -> list[int] | None:
        """Get the list of ongoing trajectories if any.

        Returns:
            Either a list trajectories or None if nothing was left to do
        """
        with self.session_scope() as session:
            stmt = select(SplittingIterations).order_by(SplittingIterations.id.desc())
            last_split = session.execute(stmt).scalars().first()
            if last_split and last_split.status == "locked":
                return last_split.discarded_traj_ids
            return None

    def get_weights(self) -> npt.NDArray[np.number]:
        """Read the weights from the database.

        Returns:
            the weight for each splitting iteration as a numpy array
        """
        with self.session_scope() as session:
            weights = session.execute(select(SplittingIterations.weight)).scalars().all()
            return np.array(weights, dtype="float32")

    def get_biases(self) -> npt.NDArray[np.number]:
        """Read the biases from the database.

        Returns:
            the bias for each splitting iteration as a numpy array
        """
        with self.session_scope() as session:
            biases = session.execute(select(SplittingIterations.bias)).scalars().all()
            return np.array(biases, dtype="int")

    def get_minmax(self) -> npt.NDArray[np.number]:
        """Read the min/max from the database.

        Returns:
            the 2D Numpy array with k_index, min, max
        """
        with self.session_scope() as session:
            stmt = select(
                SplittingIterations.split_id,
                SplittingIterations.min_max,
            )
            results = session.execute(stmt).all()
            return np.array(
                [[float(r.split_id), float(r.min_max[0]), float(r.min_max[1])] for r in results],
                dtype="float32",
            )

    def clear_splitting_data(self) -> int:
        """Delete the content of the splitting data table.

        Returns:
            The number of entries deleted
        """
        with self.session_scope() as session:
            stmt = delete(SplittingIterations)
            result = session.execute(stmt)
            return result.rowcount

    def dump_file_json(self, json_file: str | None = None) -> None:
        """Dump the content of the trajectory table to a json file.

        Args:
            json_file: an optional file name (or path) to dump the data to
        """
        db_data: dict[str, Any] = {}
        with self.session_scope() as session:
            db_data["trajectories"] = {
                traj.id - 1: {"file": traj.traj_file, "status": traj.status, "metadata": traj.t_metadata}
                for traj in session.execute(select(Trajectory)).scalars().all()
            }
            db_data["archived_trajectories"] = {
                traj.id - 1: {"file": traj.traj_file, "metadata": traj.t_metadata}
                for traj in session.execute(select(ArchivedTrajectory)).scalars().all()
            }
            splits = session.execute(select(SplittingIterations)).scalars().all()
            db_data["splitting_data"] = {
                s.id: {
                    "k": s.split_id,
                    "bias": s.bias,
                    "weight": s.weight,
                    "min_max_start": s.min_max,
                    "discarded_ids": s.discarded_traj_ids,
                    "ancestor_ids": s.ancestor_traj_ids,
                    "min_vals": s.min_vals,
                    "status": s.status,
                }
                for s in splits
            }

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
