"""A set of functions used by TAMS workers."""

import asyncio
import concurrent.futures
import datetime
import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from pytams.database import Database
from pytams.trajectory import Trajectory
from pytams.trajectory import WallTimeLimitError

_logger = logging.getLogger(__name__)


def traj_advance_with_exception(traj: Trajectory, walltime: float, a_db: Database | None) -> Trajectory:
    """Advance a trajectory with exception handling.

    Args:
        traj: a trajectory
        walltime: the time limit to advance the trajectory
        a_db: a database

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
        if a_db:
            a_db.unlock_trajectory(traj.id(), traj.has_ended())
            a_db.save_trajectory(traj)

    return traj


def pool_worker(traj: Trajectory, end_date: datetime.date, db_path: str | None = None) -> Trajectory:
    """A worker to generate each initial trajectory.

    Args:
        traj: a trajectory
        end_date: the time limit to advance the trajectory
        db_path: a path to a TAMS database or None

    Returns:
        The updated trajectory
    """
    # Get wall time
    wall_time = -1.0
    timedelta: datetime.timedelta = end_date - datetime.datetime.now(tz=datetime.timezone.utc)
    if timedelta:
        wall_time = timedelta.total_seconds()

    if wall_time > 0.0 and not traj.has_ended():
        db = None
        if db_path:
            db = Database.load(Path(db_path))
            # Try to lock the trajectory in the DB
            get_to_work = db.lock_trajectory(traj.id())
            if not get_to_work:
                return traj

        inf_msg = f"Advancing {traj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        traj = traj_advance_with_exception(traj, wall_time, db)

    return traj


def ms_worker(
    from_traj: Trajectory,
    rst_traj: Trajectory,
    min_val: float,
    end_date: datetime.date,
    db_path: str | None = None,
) -> Trajectory:
    """A worker to restart trajectories.

    Args:
        from_traj: a trajectory to restart from
        rst_traj: the trajectory being restarted
        min_val: the value of the score function to restart from
        end_date: the time limit to advance the trajectory
        db_path: a database path or None
    """
    # Get wall time
    wall_time = -1.0
    timedelta: datetime.timedelta = end_date - datetime.datetime.now(tz=datetime.timezone.utc)
    if timedelta:
        wall_time = timedelta.total_seconds()

    if wall_time > 0.0:
        db = None
        if db_path:
            # Fetch a handle to the trajectory we are branching in the database pool
            # Try to lock the trajectory in the DB
            db = Database.load(Path(db_path))
            get_to_work = db.lock_trajectory(rst_traj.id(), True)
            if not get_to_work:
                err_msg = f"Unable to lock trajectory {rst_traj.id()} for branching"
                _logger.error(err_msg)
                raise RuntimeError(err_msg)

            # Archive the trajectory we are branching
            db.archive_trajectory(rst_traj)

        inf_msg = f"Restarting [{rst_traj.id()}] from {from_traj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        traj = Trajectory.branch_from_trajectory(from_traj, rst_traj, min_val)

        # The branched trajectory has a new checkfile
        # Update the database to point to the latest one.
        if db:
            db.update_trajectory_file(traj.id(), traj.get_checkfile())

        return traj_advance_with_exception(traj, wall_time, db)

    return Trajectory.branch_from_trajectory(from_traj, rst_traj, min_val)


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
