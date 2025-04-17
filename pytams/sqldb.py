"""A class for the TAMS data as an SQL database using SQLAlchemy."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

_logger = logging.getLogger(__name__)

Base = declarative_base()

class Trajectory(Base):
    """A table storing the trajectories."""
    __tablename__ = 'trajectories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    traj_file = Column(Text, nullable=False)
    status = Column(String, default="idle", nullable=False)

class ArchivedTrajectory(Base):
    """A table storing the archived trajectories."""
    __tablename__ = 'archived_trajectories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    traj_file = Column(Text, nullable=False)


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
    def __init__(self, file_name: str) -> None:
        """Initialize the file.

        Args:
            file_name : The file name
        """
        self._file_name = file_name
        self._engine = create_engine(f"sqlite:///{file_name}", echo=False)
        self._Session = sessionmaker(bind=self._engine)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the tables of the file.

        Raises:
            RuntimeError : If a connection to the DB could not be acquired
        """
        try:
            Base.metadata.create_all(self._engine)
        except SQLAlchemyError as e:
            _logger.exception("Failed to initialize DB schema")
            raise RuntimeError("Database initialization failed") from e

    def add_trajectory(self, traj_file: str) -> None:
        """Add a new trajectory to the DB.

        Args:
            traj_file : The trajectory file of that trajectory

        Raises:
            SQLAlchemyError if the DB could not be accessed
        """
        session = self._Session()
        try:
            new_traj = Trajectory(traj_file=traj_file)
            session.add(new_traj)
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to add trajectory")
            raise
        finally:
            session.close()

    def lock_trajectory(self,
                        traj_id: int,
                        allow_completed_lock: bool = False) -> bool:
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
            else:
                err_msg = f"Trajectory {traj_id} does not exist"
                _logger.error(err_msg)
                raise ValueError(err_msg)
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to lock trajectory")
            raise
        finally:
            session.close()

    def mark_trajectory_as_completed(self,
                                     traj_id : int) -> None:
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

    def release_trajectory(self,
                           traj_id : int) -> None:
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

    def fetch_trajectory(self,
                         traj_id : int) -> str | None:
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
                return traj.traj_file
            else:
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

    def archive_trajectory(self,
                           traj_file : str) -> None:
        """Add a new trajectory to the archive container.

        Args:
            traj_file : The trajectory file of that trajectory
        """
        session = self._Session()
        try:
            new_traj = ArchivedTrajectory(traj_file=traj_file)
            session.add(new_traj)
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to archive trajectory")
        finally:
            session.close()

    def fetch_archived_trajectory(self,
                                  traj_id : int) -> str | None:
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
                return traj.traj_file
            else:
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

    def dump_file_json(self) -> None:
        """Dump the content of the trajectory table to a json file."""
        db_data = {}
        session = self._Session()
        try:
            db_data["trajectories"] = dict([(traj.id-1,
                                             {"file" : traj.traj_file, "status" : traj.status})
                                            for traj in session.query(Trajectory).all()])
            db_data["archived_trajectories"] = dict([(traj.id-1,
                                                      {"file" : traj.traj_file})
                                                     for traj in session.query(ArchivedTrajectory).all()])
        except SQLAlchemyError:
            session.rollback()
            _logger.exception("Failed to count the number of archived trajectories")
            raise
        finally:
            session.close()

        json_file = Path(self._file_name).parent / f"{str(Path(self._file_name).stem)}.json"
        with json_file.open("w") as f:
            json.dump(db_data, f, indent=2)
