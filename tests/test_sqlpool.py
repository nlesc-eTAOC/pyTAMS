"""Tests for the pytams.sqlpool class."""
from pathlib import Path
import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError
from pytams.sqldb import SQLFile


def test_createdb():
    """Initialize a SQLFile."""
    poolfile = SQLFile("test.db")
    assert poolfile.name() == "test.db"
    Path("./test.db").unlink(missing_ok=True)

def test_createdb_inmemory():
    """Initialize a SQLFile in memory."""
    poolfile = SQLFile("", in_memory=True)
    assert poolfile.name() == ""

def test_createdb_read_only():
    """Initialize a read only SQLFile."""
    with pytest.raises(SQLAlchemyError):
        _ = SQLFile("testRO.db", ro_mode=True)

def test_createdb_fail():
    """Fail to initialize a SQLFile."""
    with pytest.raises(SQLAlchemyError):
        _ = SQLFile("/test.db")

def test_add_traj_to_db():
    """Add a trajectory to SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    assert poolfile.get_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)

def test_add_traj_to_ro_db():
    """Try add a trajectory to an RO SQLFile."""
    poolfile = SQLFile("test.db") # First create the DB
    poolfile = SQLFile("test.db", ro_mode=True) # Open in RO
    with pytest.raises(SQLAlchemyError):
        poolfile.add_trajectory("test.xml","")
    Path("./test.db").unlink(missing_ok=True)

def test_add_traj_and_update_to_db():
    """Add and update a trajectory to SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    assert poolfile.fetch_trajectory(0) == "test.xml"
    poolfile.update_trajectory_file(0, "UpdatedTest.xml")
    assert poolfile.fetch_trajectory(0) == "UpdatedTest.xml"
    Path("./test.db").unlink(missing_ok=True)

def test_try_update_traj_to_db():
    """Try update missing trajectory to SQLFile."""
    poolfile = SQLFile("test.db")
    with pytest.raises(SQLAlchemyError):
        poolfile.update_trajectory_file(0, "UpdatedTest.xml")
    Path("./test.db").unlink(missing_ok=True)

def test_add_traj_to_db_inmemory():
    """Add a trajectory to SQL database in memory."""
    poolfile = SQLFile("", in_memory=True)
    poolfile.add_trajectory("test.xml","")
    assert poolfile.get_trajectory_count() == 1

def test_add_traj_to_missing_db():
    """Add a trajectory to a deleted SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    assert poolfile.get_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)
    with pytest.raises(SQLAlchemyError):
        poolfile.add_trajectory("test2.xml","")

def test_archive_traj_to_db():
    """Archive a trajectory to SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.archive_trajectory("test.xml","")
    assert poolfile.get_archived_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)

def test_add_traj_and_fetch():
    """Add a trajectory and fetch from SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    poolfile.add_trajectory("test_2.xml","")
    assert poolfile.get_trajectory_count() == 2
    traj = poolfile.fetch_trajectory(0)
    assert traj == "test.xml"
    Path("./test.db").unlink(missing_ok=True)

def test_fetch_unknown_traj():
    """Fetch an unknown trajectory."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    assert poolfile.get_trajectory_count() == 1
    with pytest.raises(ValueError):
        _ = poolfile.fetch_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_archive_and_fetch_traj_to_db():
    """Archive a trajectory to SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.archive_trajectory("test.xml","")
    assert poolfile.get_archived_trajectory_count() == 1
    traj = poolfile.fetch_archived_trajectory(0)
    assert traj == "test.xml"
    Path("./test.db").unlink(missing_ok=True)

def test_fetch_unknown_archived_traj():
    """Fetch an unknown archived trajectory."""
    poolfile = SQLFile("test.db")
    poolfile.archive_trajectory("test.xml","")
    assert poolfile.get_archived_trajectory_count() == 1
    with pytest.raises(ValueError):
        _ = poolfile.fetch_archived_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_lock_trajectory():
    """Lock a trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    status = poolfile.lock_trajectory(0)
    assert status
    Path("./test.db").unlink(missing_ok=True)

def test_lock_locked_trajectory():
    """Lock an already locked trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    status = poolfile.lock_trajectory(0)
    status = poolfile.lock_trajectory(0)
    assert status is False
    Path("./test.db").unlink(missing_ok=True)

def test_lock_and_release_trajectory():
    """Lock and release a trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    status = poolfile.lock_trajectory(0)
    poolfile.release_trajectory(0)
    status = poolfile.lock_trajectory(0)
    assert status
    Path("./test.db").unlink(missing_ok=True)

def test_lock_and_complete_trajectory():
    """Lock and mark complete a trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    _ = poolfile.lock_trajectory(0)
    poolfile.mark_trajectory_as_completed(0)
    Path("./test.db").unlink(missing_ok=True)

def test_lock_and_complete_unknown_trajectory():
    """Lock and try to mark a trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    _ = poolfile.lock_trajectory(0)
    with pytest.raises(ValueError):
        poolfile.mark_trajectory_as_completed(1)
    Path("./test.db").unlink(missing_ok=True)

def test_lock_and_release_multiple_trajectory():
    """Lock and release several trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    for i in range(10):
        poolfile.add_trajectory(f"test{i}.xml","")
        status = poolfile.lock_trajectory(0)
    poolfile.release_all_trajectories()
    status = True
    for _ in range(10):
        status = status or poolfile.lock_trajectory(0)
    assert status
    Path("./test.db").unlink(missing_ok=True)

def test_lock_unknown_trajectory():
    """Lock an unknown trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    with pytest.raises(ValueError):
        _ = poolfile.lock_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_lock_in_missing_db():
    """Lock a trajectory in a missing SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    Path("./test.db").unlink(missing_ok=True)
    with pytest.raises(SQLAlchemyError):
        _ = poolfile.lock_trajectory(0)

def test_release_unknown_trajectory():
    """Release an unknown trajectory in the SQLFile."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    with pytest.raises(ValueError):
        poolfile.release_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_splitting_data_add():
    """Adding splitting data to the database."""
    poolfile = SQLFile("test.db")
    for i in range(10):
        poolfile.add_splitting_data(i, 1, 0.1, [i-1], [0], [0.0], [0.0, 0.0])
        poolfile.mark_last_iteration_as_completed()
    Path("./test.db").unlink(missing_ok=True)

def test_splitting_data_add_and_ongoing():
    """Adding splitting data to the database."""
    poolfile = SQLFile("", in_memory=True)
    poolfile.mark_last_iteration_as_completed()
    for i in range(10):
        poolfile.add_splitting_data(i, 1, 0.1, [i-1], [0], [0.0], [0.0, 0.0])
        poolfile.mark_last_iteration_as_completed()
    assert poolfile.get_ongoing() is None
    poolfile.add_splitting_data(10, 1, 0.1, [10-1,1,56], [0], [0.0], [0.0, 0.0])
    assert poolfile.get_ongoing() == [9,1,56]

def test_splitting_data_add_and_query():
    """Adding splitting data to the database."""
    poolfile = SQLFile("", in_memory=True)
    for i in range(1):
        poolfile.add_splitting_data(2*i, 1, 0.1, [2*i-1], [0], [0.0], [0.0, 0.0])
        poolfile.mark_last_iteration_as_completed()
    assert poolfile.get_minmax().all() == np.array([2.0,0.0,0.0]).all()

def test_splitting_data_query_fail():
    """Adding splitting data to the database."""
    poolfile = SQLFile("test.db")
    for i in range(1):
        poolfile.add_splitting_data(2*i, 1, 0.1, [2*i-1], [0], [0.0], [0.0, 0.0])
    assert poolfile.get_k_split() == 1

    poolfile = SQLFile("test.db", ro_mode=True)
    with pytest.raises(SQLAlchemyError):
        poolfile.mark_last_iteration_as_completed()
    Path("./test.db").unlink(missing_ok=True)

def test_dump_json():
    """Dump the content of the DB to a json file."""
    poolfile = SQLFile("test.db")
    poolfile.add_trajectory("test.xml","")
    poolfile.archive_trajectory("test_arch.xml","")
    poolfile.dump_file_json()
    assert Path("./test.json").exists() is True
    Path("./test.db").unlink(missing_ok=True)
    Path("./test.json").unlink(missing_ok=True)
