"""A class for the TAMS data as an SQL database using SQLAlchemy."""

from __future__ import annotations
import json
import logging
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import numpy as np
from sqlalchemy import Boolean
from sqlalchemy import CursorResult
from sqlalchemy import Float
from sqlalchemy import LargeBinary
from sqlalchemy import create_engine
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


class DiagnosticEntry(Base):
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


class DiagDB:
    """An SQL to keep track of the diagnostics.

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

    def add_diagnostic_entry(
        self,
        diaglabel: str,
        traj_id: int,
        level: float,
        time: float,
        weight: float,
        ldata: bytes,
    ) -> None:
        """Atomic insert of a diagnostic snapshot."""
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
        """Return the maximum level already recorded for this traj/label."""
        with self.session_scope() as session:
            # Assuming your DiagnosticEntry model has these columns
            stmt = (
                select(func.max(DiagnosticEntry.level_crossed))
                .where(DiagnosticEntry.traj_id == traj_id)
                .where(DiagnosticEntry.diaglabel == label)
            )
            result = session.scalar(stmt)
            return float(result) if result is not None else -np.inf

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

    def get_diagnostic_data(self, label: str) -> dict[float, list[tuple[Any, float]]]:
        """Retrieve all diagnostic snapshots for a specific label.

        Args:
            label: the label of the diagnostic of interest

        Returns:
            A dictionary mapping each iso-level (float) to a list of tuples.
            Each tuple contains (unpickled_data, trajectory_weight).
        """
        results_dict: dict[float, list[tuple[Any, float]]] = {}

        with self.session_scope() as session:
            # Query entries for the specific label, ordered by level
            stmt = (
                select(DiagnosticEntry.level_crossed, DiagnosticEntry.weight, DiagnosticEntry.model_data)
                .where(DiagnosticEntry.diaglabel == label)
                .order_by(DiagnosticEntry.level_crossed.asc())
            )

            rows = session.execute(stmt).all()

            for level, weight, blob in rows:
                # Unpickle the model data
                data = pickle.loads(blob)  # noqa: S301

                if level not in results_dict:
                    results_dict[level] = []

                results_dict[level].append((data, weight))

        return results_dict

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
