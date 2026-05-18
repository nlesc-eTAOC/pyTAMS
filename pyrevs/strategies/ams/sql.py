"""An extension class for the AMS data as an SQL database using SQLAlchemy."""

from __future__ import annotations
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import numpy as np
import numpy.typing as npt
from sqlalchemy import JSON
from sqlalchemy import CursorResult
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import Session
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import sessionmaker
from pyrevs.core import CoreBase

if TYPE_CHECKING:
    from collections.abc import Generator
    from sqlalchemy.engine import Engine

_logger = logging.getLogger(__name__)


class SplittingIterations(CoreBase):
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


class AMSDB:
    """An extension of the core database holding AMS iterations repertoire.

    Allows atomic access to an SQL database from all
    the workers.

    Attributes:
        _file_name : The file name
    """

    def __init__(self, engine: Engine) -> None:
        """Initialize the file.

        Args:
            engine : The SQLAlchemy engine
        """
        self._engine = engine
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._file_name: str = engine.url.database if engine.url.database else "ams.db"

        CoreBase.metadata.create_all(self._engine)

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
        """Get the current discarded trajectories counter.

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

        When running pyREVS with AMS, at each new iteration the ensemble minimum
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
            return np.array(weights, dtype="float64")

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
                dtype="float64",
            )

    def clear_splitting_data(self) -> int:
        """Delete the content of the splitting data table.

        Returns:
            The number of entries deleted
        """
        with self.session_scope() as session:
            stmt = delete(SplittingIterations)
            result = session.execute(stmt)
            return int(cast("CursorResult", result).rowcount or 0)

    def dump_file_json(self, json_file: str | None = None) -> None:
        """Dump the content of the trajectory table to a json file.

        Args:
            json_file: an optional file name (or path) to dump the data to
        """
        db_data: dict[str, Any] = {}
        with self.session_scope() as session:
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
