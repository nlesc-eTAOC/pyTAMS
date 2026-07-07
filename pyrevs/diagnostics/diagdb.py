"""A class for the pyREVS diagnostics data as an SQL database using SQLAlchemy."""

from __future__ import annotations
import json
import logging
import pickle
from pathlib import Path
from typing import Any
from typing import cast
import numpy as np
from sqlalchemy import Boolean
from sqlalchemy import CursorResult
from sqlalchemy import Float
from sqlalchemy import LargeBinary
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from pyrevs.core.sqlmanager import BaseSQLManager

_logger = logging.getLogger(__name__)


class DiagBase(DeclarativeBase):
    """A base class for the tables."""


class DiagnosticEntry(DiagBase):
    """Table for recording model data at specific score levels."""

    __tablename__ = "diagnostics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    traj_id: Mapped[int] = mapped_column(nullable=False)
    level_crossed: Mapped[float] = mapped_column(Float, nullable=False)
    time: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    model_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    diaglabel: Mapped[str] = mapped_column(nullable=False)


class DiagDB(BaseSQLManager):
    """A database to keep track of the diagnostics data.

    Diagnostic entries are agregated in single table. Each entry
    is associated to a trajectory and its weight in the ensemble.
    """

    @classmethod
    def default_name(cls) -> str:
        """Default name for the database file."""
        return "diagDB.db"

    def __init__(self, file_name: str | None = None, in_memory: bool = False, ro_mode: bool = False) -> None:
        """Initialize the file.

        Args:
            file_name : The file name
            in_memory: a bool to trigger in-memory creation
            ro_mode: a bool to trigger read-only access to the database
        """
        if file_name is None:
            file_name = self.default_name()
        super().__init__(file_name, DiagBase.metadata, in_memory, ro_mode)

    def add_diagnostic_entry(
        self,
        diaglabel: str,
        traj_id: int,
        level: float,
        time: float,
        weight: float,
        ldata: bytes,
    ) -> None:
        """Atomic insert of a diagnostic snapshot.

        The data schema assumes that any new addition to the database is
        made on an active trajectory.

        Args:
            diaglabel: the label of the diagnostic inserting the entry
            traj_id: the ID of the traj adding the entry
            level: the score level of the entry
            time: the trajectory time at which the diagnostic was triggered
            weight: the weight of the trajectory
            ldata: the actual model data stored in the database
        """
        with self.session_scope() as session:
            entry = DiagnosticEntry(
                diaglabel=diaglabel,
                traj_id=traj_id,
                level_crossed=level,
                time=time,
                weight=weight,
                active=True,
                model_data=ldata,
            )
            session.add(entry)

    def get_highest_recorded_level(self, traj_id: int, label: str) -> float:
        """Return the maximum level already recorded for this traj/label.

        Args:
            traj_id: the ID of a trajectory
            label: the label of the diagnostic targeter

        Returns:
            the highest value of level_crossed
        """
        with self.session_scope() as session:
            stmt = (
                select(func.max(DiagnosticEntry.level_crossed))
                .where(DiagnosticEntry.traj_id == traj_id)
                .where(DiagnosticEntry.diaglabel == label)
            )
            result = session.scalar(stmt)
            return float(result) if result is not None else -np.inf

    def get_last_diagnostic_entry_metadata(self, traj_id: int, label: str) -> tuple[float, float, float] | None:
        """Return the last diagnostic entry metadata for a given trajectory.

        Args:
            traj_id: the ID of a trajectory
            label: the label of the diagnostic targeter

        Returns:
            a tuple (level, time, weight)
        """
        with self.session_scope() as session:
            stmt = (
                select(
                    DiagnosticEntry.level_crossed,
                    DiagnosticEntry.time,
                    DiagnosticEntry.weight,
                )
                .where(DiagnosticEntry.traj_id == traj_id)
                .where(DiagnosticEntry.diaglabel == label)
                .order_by(DiagnosticEntry.id.desc())
                .limit(1)
            )
            result = session.execute(stmt).first()
            return cast("tuple[float, float, float]", result) if result is not None else None

    def duplicate_diagnostic_history(
        self,
        ancestor_id: int,
        discarded_id: int,
        new_id: int,
        new_weight: float,
        threshold: float,
    ) -> int:
        """Copy diagnostic entries from an ancestor to a descendant.

        Copies all entries where level_crossed <= threshold.
        Returns the number of entries duplicated.

        The entries belonging to the discarded trajectory are set
        to inactive.

        Args:
            ancestor_id: the ID of the ancestor to copy data from
            discarded_id: the ID of the discarded trajectory (during sampling iterations)
            new_id: the ID of the new child trajectory
            new_weight: the weight of the new child trajectory
            threshold: the score threshold up to which copy must be performed
        """
        with self.session_scope() as session:
            # Set the discarded trajectory to inactive
            stmt_update = (
                update(DiagnosticEntry)
                .where(
                    DiagnosticEntry.traj_id == discarded_id,
                )
                .values(active=False)
            )
            session.execute(stmt_update)

            # Select the relevant entries from the ancestor
            # Fetched as dictionaries to easily modify them for insertion
            stmt = select(DiagnosticEntry).where(
                DiagnosticEntry.traj_id == ancestor_id, DiagnosticEntry.level_crossed <= threshold
            )
            ancestor_entries = session.execute(stmt).scalars().all()

            if not ancestor_entries:
                return 0

            new_entries = []
            for entry in ancestor_entries:
                # Create a new entry object (stripping the original primary key 'id')
                new_entry = DiagnosticEntry(
                    diaglabel=entry.diaglabel,
                    traj_id=new_id,
                    level_crossed=entry.level_crossed,
                    time=entry.time,
                    weight=new_weight,
                    active=True,
                    model_data=entry.model_data,
                )
                new_entries.append(new_entry)

            session.add_all(new_entries)

            return len(new_entries)

    def duplicate_diagnostic_history_from_time(
        self,
        ancestor_id: int,
        discarded_id: int,
        new_id: int,
        new_weight: float,
        branching_time: float,
    ) -> int:
        """Copy diagnostic entries from an ancestor to a descendant.

        Copies all entries where time <= branching_time.
        Returns the number of entries duplicated.

        The entries belonging to the discarded trajectory are set
        to inactive.

        Args:
            ancestor_id: the ID of the ancestor to copy data from
            discarded_id: the ID of the discarded trajectory (during sampling iterations)
            new_id: the ID of the new child trajectory
            new_weight: the weight of the new child trajectory
            branching_time: the time up to which copy must be performed
        """
        with self.session_scope() as session:
            # Set the discarded trajectory to inactive
            stmt_update = (
                update(DiagnosticEntry)
                .where(
                    DiagnosticEntry.traj_id == discarded_id,
                )
                .values(active=False)
            )
            session.execute(stmt_update)

            # Select the relevant entries from the ancestor
            # Fetched as dictionaries to easily modify them for insertion
            stmt = select(DiagnosticEntry).where(
                DiagnosticEntry.traj_id == ancestor_id, DiagnosticEntry.time <= branching_time
            )
            ancestor_entries = session.execute(stmt).scalars().all()

            if not ancestor_entries:
                return 0

            new_entries = []
            for entry in ancestor_entries:
                # Create a new entry object (stripping the original primary key 'id')
                new_entry = DiagnosticEntry(
                    diaglabel=entry.diaglabel,
                    traj_id=new_id,
                    level_crossed=entry.level_crossed,
                    time=entry.time,
                    weight=new_weight,
                    active=True,
                    model_data=entry.model_data,
                )
                new_entries.append(new_entry)

            session.add_all(new_entries)

            return len(new_entries)

    def update_all_active_weights(self, new_weight: float) -> int:
        """Update all the active trajectories weight.

        Args:
            new_weight: the updated weight

        Returns:
            the number of trajectory updated
        """
        with self.session_scope() as session:
            stmt_update = (
                update(DiagnosticEntry)
                .where(
                    DiagnosticEntry.active,
                )
                .values(weight=new_weight)
            )
            result = session.execute(stmt_update)
            return int(cast("CursorResult", result).rowcount or 0)

    def get_diagnostic_data(self, label: str) -> dict[float, list[tuple[Any, float, float, int]]]:
        """Retrieve all diagnostic snapshots for a specific label.

        Args:
            label: the label of the diagnostic of interest

        Returns:
            A dictionary mapping each iso-level (float) to a list of tuples.
            Each tuple contains (unpickled_data, trajectory_weight, time, tid).
        """
        results_dict: dict[float, list[tuple[Any, float, float, int]]] = {}

        with self.session_scope() as session:
            # Query entries for the specific label, ordered by level
            stmt = (
                select(
                    DiagnosticEntry.level_crossed,
                    DiagnosticEntry.weight,
                    DiagnosticEntry.time,
                    DiagnosticEntry.traj_id,
                    DiagnosticEntry.model_data,
                )
                .where(DiagnosticEntry.diaglabel == label)
                .order_by(DiagnosticEntry.level_crossed.asc())
            )

            rows = session.execute(stmt).all()

            for level, weight, time, tid, blob in rows:
                # Unpickle the model data
                data = pickle.loads(blob)  # noqa: S301

                if level not in results_dict:
                    results_dict[level] = []

                results_dict[level].append((data, weight, time, tid))

        return results_dict

    def get_diagnostic_data_traj(
        self, label: str, tid: int, time_ordered: bool = False
    ) -> dict[float, list[tuple[Any, float, float]]]:
        """Retrieve diagnostic snapshots for a specific label and trajectory.

        Args:
            label: the label of the diagnostic of interest
            tid: the ID of the trajectory
            time_ordered: whether to order the results by time (default: False, by level)

        Returns:
            A dictionary mapping each iso-level (float) to a list of tuples.
            Each tuple contains (unpickled_data, trajectory_weight, time, tid).
        """
        results_dict: dict[float, list[tuple[Any, float, float]]] = {}

        with self.session_scope() as session:
            # Query entries for the specific label, ordered by time or level
            if time_ordered:
                stmt = (
                    select(
                        DiagnosticEntry.level_crossed,
                        DiagnosticEntry.weight,
                        DiagnosticEntry.time,
                        DiagnosticEntry.model_data,
                    )
                    .where(DiagnosticEntry.diaglabel == label)
                    .where(DiagnosticEntry.traj_id == tid)
                    .order_by(DiagnosticEntry.time.asc())
                )
            else:
                stmt = (
                    select(
                        DiagnosticEntry.level_crossed,
                        DiagnosticEntry.weight,
                        DiagnosticEntry.time,
                        DiagnosticEntry.model_data,
                    )
                    .where(DiagnosticEntry.diaglabel == label)
                    .where(DiagnosticEntry.traj_id == tid)
                    .order_by(DiagnosticEntry.level_crossed.asc())
                )

            rows = session.execute(stmt).all()

            for level, weight, time, blob in rows:
                # Unpickle the model data
                data = pickle.loads(blob)  # noqa: S301

                if level not in results_dict:
                    results_dict[level] = []

                results_dict[level].append((data, weight, time))

        return results_dict

    def delete_traj_diagnostic_data(self, traj_id: int) -> None:
        """Delete all the diagnostic data for a specific trajectory.

        Args:
            traj_id: the ID of the trajectory
        """
        with self.session_scope() as session:
            stmt = delete(DiagnosticEntry).where(DiagnosticEntry.traj_id == traj_id)
            session.execute(stmt)

    def get_unique_traj_ids(self) -> list[int]:
        """Return the list of unique trajectory IDs."""
        with self.session_scope() as session:
            stmt = (
                select(DiagnosticEntry.traj_id)
                .distinct()
                .order_by(DiagnosticEntry.traj_id)
            )

            return list(session.scalars(stmt))

    def count_entries(self) -> int:
        """Return the total number of rows in the diagnostics table."""
        with self.session_scope() as session:
            stmt = select(func.count()).select_from(DiagnosticEntry)
            return session.scalar(stmt)

    def dump_to_json(self, json_path: str) -> None:
        """Export the entire diagnostic database to a JSON file.

        Note that the content of the data stored in the database is
        omitted. Only the metadata of each stored data is dumpe for debuggin purposes.
        """
        dump_data = []

        with self.session_scope() as session:
            # Fetch every entry in the database
            stmt = select(DiagnosticEntry).order_by(DiagnosticEntry.traj_id, DiagnosticEntry.level_crossed)
            results = session.execute(stmt).scalars().all()

            for entry in results:
                # Prepare the row dictionary
                row = {
                    "diaglabel": entry.diaglabel,
                    "traj_id": entry.traj_id,
                    "level_crossed": float(entry.level_crossed),
                    "time": float(entry.time),
                    "weight": float(entry.weight),
                    "active": entry.active,
                }
                dump_data.append(row)

        # Write to file with pretty printing
        with Path(json_path).open("w") as f:
            json.dump(dump_data, f, indent=4)

    def close(self) -> None:
        """Dispose of the engine and clear connections."""
        if self._engine:
            self._engine.dispose()
