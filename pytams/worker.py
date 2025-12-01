"""A set of functions used by TAMS workers."""

import asyncio
import concurrent.futures
import datetime
import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from pytams.sqldb import SQLFile
from pytams.trajectory import Trajectory
from pytams.trajectory import WallTimeLimitError

_logger = logging.getLogger(__name__)


def update_trajectory_in_sql(traj: Trajectory, sqldb: SQLFile | None = None, db_path: str | None = None) -> None:
    """Wrapper for update SQL trajectory info.

    Args:
        sqldb: the SQL database to update
        traj: the traj to get the information from
        db_path: an optional TAMS database path
    """
    if sqldb:
        checkfile_str = (
            traj.get_checkfile().relative_to(Path(db_path)).as_posix() if db_path else traj.get_checkfile().as_posix()
        )
        sqldb.update_trajectory(traj.id(), checkfile_str, traj.serialize_metadata_json())


def traj_advance_with_exception(
    traj: Trajectory, walltime: float, sqldb: SQLFile | None = None, db_path: str | None = None
) -> Trajectory:
    """Advance a trajectory with exception handling.

    Args:
        traj: a trajectory
        walltime: the time limit to advance the trajectory
        sqldb: a handle to the SQL database
        db_path: an optional path to the run database

    Returns:
        The updated trajectory
    """
    try:
        traj.advance(walltime=walltime)

    except WallTimeLimitError:
        warn_msg = f"Trajectory {traj.idstr()} advance ran out of time !"
        _logger.warning(warn_msg)

    except Exception:
        err_msg = f"Trajectory {traj.idstr()} advance ran into an error !"
        _logger.exception(err_msg)
        raise

    finally:
        # Update the SQL database
        if sqldb:
            if traj.has_ended():
                sqldb.mark_trajectory_as_completed(traj.id())
            else:
                sqldb.release_trajectory(traj.id())
            update_trajectory_in_sql(traj, sqldb, db_path)

        # Trigger a checkfile dump if we are provided with
        # a database path
        if db_path:
            traj.store()

    return traj


def pool_worker(
    traj: Trajectory, end_date: datetime.date, sql_path: str | None = None, db_path: str | None = None
) -> Trajectory:
    """A worker to generate each initial trajectory.

    Args:
        traj: a trajectory
        end_date: the time limit to advance the trajectory
        sql_path: an optional path to the SQL database
        db_path: an optional path to the run database

    Returns:
        The updated trajectory
    """
    # Get wall time
    wall_time = -1.0
    timedelta: datetime.timedelta = end_date - datetime.datetime.now(tz=datetime.timezone.utc)
    if timedelta:
        wall_time = timedelta.total_seconds()

    if wall_time > 0.0 and not traj.has_ended():
        # Try to lock the trajectory in the DB
        sqldb = None
        if sql_path:
            sqldb = SQLFile(sql_path)
            get_to_work = sqldb.lock_trajectory(traj.id(), allow_completed_lock=True)
            if not get_to_work:
                return traj

        inf_msg = f"Advancing {traj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        traj = traj_advance_with_exception(traj, wall_time, sqldb, db_path)

    return traj


def ms_worker(
    from_traj: Trajectory,
    rst_traj: Trajectory,
    min_val: float,
    new_weight: float,
    end_date: datetime.date,
    sql_path: str | None = None,
    db_path: str | None = None,
) -> Trajectory:
    """A worker to restart trajectories.

    Args:
        from_traj: a trajectory to restart from
        rst_traj: the trajectory being restarted
        min_val: the value of the score function to restart from
        new_weight: the weight of the new child trajectory
        end_date: the time limit to advance the trajectory
        sql_path: a path to the SQL database
        db_path: an optional path to the run database
    """
    # Get wall time
    wall_time = -1.0
    timedelta: datetime.timedelta = end_date - datetime.datetime.now(tz=datetime.timezone.utc)
    if timedelta:
        wall_time = timedelta.total_seconds()

    sqldb = None
    if sql_path:
        sqldb = SQLFile(sql_path)

    if wall_time > 0.0:
        # Try to lock the trajectory in the DB
        if sqldb:
            get_to_work = sqldb.lock_trajectory(rst_traj.id(), allow_completed_lock=True)
            if not get_to_work:
                err_msg = f"Unable to lock trajectory {rst_traj.id()} for branching"
                _logger.error(err_msg)
                raise RuntimeError(err_msg)

        inf_msg = f"Restarting [{rst_traj.id()}] from {from_traj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        traj = Trajectory.branch_from_trajectory(from_traj, rst_traj, min_val, new_weight)

        # The branched trajectory has a new checkfile
        # Update the database to point to the latest one.
        update_trajectory_in_sql(traj, sqldb, db_path)

        return traj_advance_with_exception(traj, wall_time, sqldb, db_path)

    traj = Trajectory.branch_from_trajectory(from_traj, rst_traj, min_val, new_weight)

    warn_msg = "MS worker ran out of time before advancing trajectory!"
    _logger.warning(warn_msg)

    # The branched trajectory has a new checkfile, even if haven't advanced yet
    # Update the database to point to the latest one.
    update_trajectory_in_sql(traj, sqldb, db_path)

    return traj


async def worker_async(
    queue: asyncio.Queue[tuple[Callable[..., Any], Trajectory, float, bool, str]],
    res_queue: asyncio.Queue[asyncio.Future[Trajectory]],
    executor: concurrent.futures.Executor,
) -> None:
    """An async worker for the asyncio taskrunner.

    It wraps the call to one of the above worker functions
    with access to the queue.

    Args:
        queue: a queue from which to get tasks
        res_queue: a queue to put the results in
        executor: an executor to launch the work in
    """
    while True:
        func, *work_unit = await queue.get()
        loop = asyncio.get_running_loop()
        traj: asyncio.Future[Trajectory] = await loop.run_in_executor(
            executor,
            functools.partial(func, *work_unit),
        )
        await res_queue.put(traj)
        queue.task_done()
