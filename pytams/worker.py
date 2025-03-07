"""A set of functions used by TAMS workers."""
import asyncio
import concurrent.futures
import functools
import logging
import time
from typing import Tuple
from pytams.sqldb import SQLFile
from pytams.trajectory import Trajectory
from pytams.trajectory import WallTimeLimit

_logger = logging.getLogger(__name__)

def traj_advance_with_exception(traj: Trajectory,
                                wall_time: float,
                                saveDB: bool,
                                nameDB: str) -> Trajectory:
    """Advance a trajectory with exception handling.

    Args:
        traj: a trajectory
        wall_time: the time limit to advance the trajectory
        saveDB: a bool to save the trajectory to database
        nameDB: name of the database

    Returns:
        The updated trajectory
    """
    try:
        traj.advance(walltime=wall_time)

    except WallTimeLimit:
        warn_msg = f"Trajectory {traj.idstr()} advance ran out of time !"
        _logger.warning(warn_msg)

    except Exception:
        err_msg = f"Trajectory {traj.idstr()} advance ran into an error !"
        _logger.error(err_msg)
        raise

    finally:
        if saveDB:
            pool_file = f"./{nameDB}/trajPool.db"
            sqlpool = SQLFile(pool_file)
            traj.store()
            if traj.hasEnded():
                sqlpool.mark_trajectory_as_completed(traj.id())
            else:
                sqlpool.release_trajectory(traj.id())

    return traj


def pool_worker(traj: Trajectory,
                wall_time_info: float,
                saveDB: bool,
                nameDB: str) -> Trajectory:
    """A worker to generate each initial trajectory.

    Args:
        traj: a trajectory
        wall_time_info: the time limit to advance the trajectory
        saveDB: a bool to save the trajectory to database
        nameDB: name of the database

    Returns:
        The updated trajectory
    """
    # Get wall time
    wall_time = wall_time_info - time.monotonic()

    if wall_time > 0.0 and not traj.hasEnded():
        # Fetch a handle to the trajectory in the database pool
        # Try to lock the trajectory
        if saveDB:
            pool_file = f"./{nameDB}/trajPool.db"
            sqlpool = SQLFile(pool_file)
            get_to_work = sqlpool.lock_trajectory(traj.id())
            if not get_to_work:
                return traj

        inf_msg = f"Advancing {traj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        return traj_advance_with_exception(traj, wall_time, saveDB, nameDB)

    return traj


async def worker_async(
    queue : asyncio.Queue[Tuple[Trajectory, float, bool, str]],
    res_queue : asyncio.Queue[Trajectory],
    executor : concurrent.futures.Executor) -> None:
    """A worker to generate each initial trajectory.

    Args:
        queue: a queue from which to get tasks
        res_queue: a queue to put the results in
        executor: an executor to launch the work in
    """
    while True:
        func, *work_unit = await queue.get()
        loop = asyncio.get_running_loop()
        traj = await loop.run_in_executor(
            executor,
            functools.partial(func, *work_unit)
        )
        await res_queue.put(traj)
        queue.task_done()

def ms_worker(
    t_end: float,
    fromTraj: Trajectory,
    rstId: int,
    min_val: float,
    wall_time_info: float,
    saveDB: bool,
    nameDB: str,
) -> Trajectory:
    """A worker to restart trajectories.

    Args:
        t_end: a final time
        fromTraj: a trajectory to restart from
        rstId: Id of the trajectory being worked on
        min_val: the value of the score function to restart from
        wall_time_info: the time limit to advance the trajectory
        saveDB: a bool to save the trajectory to database
        nameDB: name of DB to save the traj in (Opt)
    """
    # Get wall time
    wall_time = wall_time_info - time.monotonic()

    if wall_time > 0.0 :
        # Fetch a handle to the trajectory we are branching in the database pool
        # Try to lock the trajectory
        if saveDB:
            pool_file = f"./{nameDB}/trajPool.db"
            sqlpool = SQLFile(pool_file)
            get_to_work = sqlpool.lock_trajectory(rstId, True)
            if not get_to_work:
                err_msg = f"Unable to lock trajectory {rstId} for branching"
                _logger.error(err_msg)
                raise RuntimeError(err_msg)

        inf_msg = f"Restarting [{rstId}] from {fromTraj.idstr()} [time left: {wall_time}]"
        _logger.info(inf_msg)

        traj = Trajectory.restartFromTraj(fromTraj, rstId, min_val)

        if saveDB:
            traj.setCheckFile(f"{nameDB}/trajectories/{traj.idstr()}.xml")

        return traj_advance_with_exception(traj, wall_time, saveDB, nameDB)

    return Trajectory.restartFromTraj(fromTraj, rstId, min_val)
