"""Tests for the pytams.sqlpool class."""
from pathlib import Path
import pytest
from pytams.sqldb import SQLFile


def test_createDB():
    """Initialize a SQLFile."""
    poolFile = SQLFile("test.db")
    assert poolFile._file_name == "test.db"
    Path("./test.db").unlink(missing_ok=True)

def test_add_traj_to_DB():
    """Add a trajectory to SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    assert poolFile.get_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)

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

def test_lock_unknown_trajectory():
    """Lock an unknown trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    with pytest.raises(ValueError):
        _ = poolFile.lock_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)

def test_release_unknown_trajectory():
    """Release an unknown trajectory in the SQLFile."""
    poolFile = SQLFile("test.db")
    poolFile.add_trajectory("test.xml")
    with pytest.raises(ValueError):
        poolFile.release_trajectory(1)
    Path("./test.db").unlink(missing_ok=True)
