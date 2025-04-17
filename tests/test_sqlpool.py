"""Tests for the pytams.sqlpool class."""
from pathlib import Path
import pytest
from sqlalchemy.exc import SQLAlchemyError
from pytams.sqldb import SQLFile


def test_createDB():
    """Initialize a SQLFile."""
    poolFile = SQLFile("test.db")
    assert poolFile._file_name == "test.db"
    Path("./test.db").unlink(missing_ok=True)

def test_createDBFail():
    """Fail to initialize a SQLFile."""
    with pytest.raises(RuntimeError):
        _ = SQLFile("/test.db")

def test_add_traj_to_DB():
    """Add a trajectory to SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    assert poolFile.get_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)

def test_add_traj_to_missing_DB():
    """Add a trajectory to a deleted SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    assert poolFile.get_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)
    with pytest.raises(SQLAlchemyError):
        poolFile.add_trajectory("test2.xml")

def test_archive_traj_to_DB():
    """Archive a trajectory to SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.archive_trajectory("test.xml")
    assert poolFile.get_archived_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)

def test_add_traj_and_fetch():
    """Add a trajectory and fetch from SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    poolFile.add_trajectory("test_2.xml")
    assert poolFile.get_trajectory_count() == 2
    traj = poolFile.fetch_trajectory(0)
    assert traj == "test.xml"
    Path("./test.db").unlink(missing_ok=True)

def test_fetch_unknown_traj():
    """Fetch an unknown trajectory."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    assert poolFile.get_trajectory_count() == 1
    with pytest.raises(ValueError):
        _ = poolFile.fetch_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_archive_and_fetch_traj_to_DB():
    """Archive a trajectory to SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.archive_trajectory("test.xml")
    assert poolFile.get_archived_trajectory_count() == 1
    traj = poolFile.fetch_archived_trajectory(0)
    assert traj == "test.xml"
    Path("./test.db").unlink(missing_ok=True)

def test_fetch_unknown_archived_traj():
    """Fetch an unknown archived trajectory."""
    poolFile = SQLFile("test.db")
    poolFile.archive_trajectory("test.xml")
    assert poolFile.get_archived_trajectory_count() == 1
    with pytest.raises(ValueError):
        _ = poolFile.fetch_archived_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_lock_trajectory():
    """Lock a trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    status = poolFile.lock_trajectory(0)
    assert status
    Path("./test.db").unlink(missing_ok=True)

def test_lock_locked_trajectory():
    """Lock an already locked trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    status = poolFile.lock_trajectory(0)
    status = poolFile.lock_trajectory(0)
    assert status is False
    Path("./test.db").unlink(missing_ok=True)

def test_lock_and_release_trajectory():
    """Lock and release a trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    status = poolFile.lock_trajectory(0)
    poolFile.release_trajectory(0)
    status = poolFile.lock_trajectory(0)
    assert status
    Path("./test.db").unlink(missing_ok=True)

def test_lock_and_release_multiple_trajectory():
    """Lock and release several trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    for i in range(10):
        poolFile.add_trajectory(f"test{i}.xml")
        status = poolFile.lock_trajectory(0)
    poolFile.release_all_trajectories()
    status = True
    for i in range(10):
        status = status or poolFile.lock_trajectory(0)
    assert status
    Path("./test.db").unlink(missing_ok=True)

def test_lock_unknown_trajectory():
    """Lock an unknown trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    with pytest.raises(ValueError):
        _ = poolFile.lock_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_lock_in_missing_DB():
    """Lock a trajectory in a missing SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    Path("./test.db").unlink(missing_ok=True)
    with pytest.raises(SQLAlchemyError):
        _ = poolFile.lock_trajectory(0)

def test_release_unknown_trajectory():
    """Release an unknown trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    with pytest.raises(ValueError):
        poolFile.release_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_dump_json():
    """Dump the content of the DB to a json file."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    poolFile.archive_trajectory("test_arch.xml")
    poolFile.dump_file_json()
    assert Path("./test.json").exists() is True
    Path("./test.db").unlink(missing_ok=True)
    Path("./test.json").unlink(missing_ok=True)
