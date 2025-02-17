"""A class for the trajectory pool as an SQL file."""
from __future__ import annotations
import logging
import sqlite3

_logger = logging.getLogger(__name__)

valid_statuses = ["locked",      # Currently being worked on
                  "idle",        # Waiting
                  "completed"]   # Finished

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
    def __init__(self,
                 file_name: str) -> None:
        """Initialize the file.

        Args:
            file_name : The file name
        """
        self._file_name = file_name
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the tables of the file.

        Raises:
            RuntimeError : If a connection to the DB could not be acquired
        """
        conn = self._connect()
        cursor = conn.cursor()

        # Create the trajectory table
        # id, file, status
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            traj_file TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'idle'
        )
        """)

        conn.commit()
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection.

        Returns:
            An sqlite3.Connection

        Raises:
            RuntimeError if no connection can be established
        """
        try:
            return sqlite3.connect(self._file_name)
        except Exception:
            err_msg = f"Unable to connect to {self._file_name}"
            _logger.exception(err_msg)
            raise


    def add_trajectory(self,
                       traj_file : str) -> None:
        """Add a new trajectory to the DB.

        Args:
            traj_file : The trajectory file of that trajectory
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO trajectories (traj_file)
        VALUES (?)
        """, (traj_file,))

        conn.commit()
        conn.close()

    def lock_trajectory(self,
                        traj_id : int,
                        allow_completed_lock : bool = False) -> bool:
        """Set the status of a trajectory to "locked" if possible.

        Args:
            traj_id : The trajectory id
            allow_completed_lock : Allow to lock a "completed" trajectory

        Return:
            True if the trajectory was successfully locked, False otherwise

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        # Use atomic transaction
        # Update only if current status allows it
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
        cursor.execute("""
        SELECT status
        FROM trajectories
        WHERE id = ?
        """, (traj_id+1,))
        traj_data = cursor.fetchone()

        if traj_data:
            status = traj_data[0]
            allowed_status = ["idle", "completed"] if allow_completed_lock else ["idle"]
            if status in allowed_status:
                cursor.execute("UPDATE trajectories SET status = ? WHERE id = ?", ("locked", traj_id+1))
                conn.commit()
                conn.close()
                return True
            else:
                conn.close()
                return False
        else:
            conn.close()
            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def mark_trajectory_as_completed(self,
                                     traj_id : int) -> None:
        """Set the status of a trajectory to "completed" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        # Use atomic transaction
        # Update only if current status allows it
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
        cursor.execute("""
        SELECT status
        FROM trajectories
        WHERE id = ?
        """, (traj_id+1,))
        traj_data = cursor.fetchone()

        if traj_data:
            status = traj_data[0]
            if status in ["locked"]:
                cursor.execute("UPDATE trajectories SET status = ? WHERE id = ?", ("completed", traj_id+1))
                conn.commit()
                conn.close()
            else:
                warn_msg = f"Attempting to mark completed Trajectory {traj_id} already in status {status}."
                _logger.warning(warn_msg)
                conn.commit()
                conn.close()
        else:
            conn.commit()
            conn.close()
            err_msg = f"Trajectory {traj_id} does not exist"
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def release_trajectory(self,
                           traj_id : int) -> None:
        """Set the status of a trajectory to "idle" if possible.

        Args:
            traj_id : The trajectory id

        Raises:
            ValueError if the trajectory with the given id does not exist
        """
        # Use atomic transaction
        # Update only if current status allows it
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
        cursor.execute("""
        SELECT status
        FROM trajectories
        WHERE id = ?
        """, (traj_id+1,))
        traj_data = cursor.fetchone()

        if traj_data:
            status = traj_data[0]
            if status in ["locked"]:
                cursor.execute("UPDATE trajectories SET status = ? WHERE id = ?", ("idle", traj_id+1))
                conn.commit()
                conn.close()
            else:
                warn_msg = f"Attempting to release {status} Trajectory {traj_id}."
                _logger.warning(warn_msg)
                conn.close()
        else:
            conn.commit()
            conn.close()
            err_msg = f"Trajectory {traj_id} does not exist."
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def get_trajectory_count(self) -> int:
        """Get the number of trajectories in the DB.

        Returns:
            The number of trajectories
        """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT() FROM trajectories")
        count = cursor.fetchone()[0]
        conn.commit()
        conn.close()

        return count

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
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
        SELECT traj_file
        FROM trajectories
        WHERE id = ?
        """, (traj_id+1,))
        traj_data = cursor.fetchone()
        if traj_data:
            conn.commit()
            conn.close()
            return traj_data[0]
        else:
            conn.commit()
            conn.close()
            err_msg = f"Trajectory {traj_id} does not exist."
            _logger.error(err_msg)
            raise ValueError(err_msg)

    def dump_file_json(self) -> None:
        """Dump the content of the trajectory table to a json file."""
        conn = self._connect()
        cursor = conn.cursor()
        trajs = cursor.execute('SELECT * FROM trajectories').fetchall()
        for t in trajs:
            print(t[0]-1, t[1], t[2])
        conn.commit()
        conn.close()
