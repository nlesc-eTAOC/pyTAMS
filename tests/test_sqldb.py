"""Tests for the pyrevs.sqlpool class."""

from pathlib import Path
import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError
from pyrevs.core import CoreDB
from pyrevs.strategies.ams.sql import AMSDB


def test_createdb():
    """Initialize a CoreDB."""
    poolfile = CoreDB("test.db")
    assert poolfile.name() == "test.db"
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_createdb_inmemory():
    """Initialize a CoreDB in memory."""
    poolfile = CoreDB("", in_memory=True)
    assert poolfile.name() == ""


def test_createdb_read_only():
    """Initialize a read only CoreDB."""
    with pytest.raises(SQLAlchemyError):
        _ = CoreDB("testRO.db", ro_mode=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_createdb_fail():
    """Fail to initialize a CoreDB."""
    with pytest.raises(SQLAlchemyError):
        _ = CoreDB("/test.db")


def test_add_traj_to_db():
    """Add a trajectory to CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    assert poolfile.get_trajectory_count() == 1
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_add_traj_to_ro_db():
    """Try add a trajectory to an RO CoreDB."""
    poolfile = CoreDB("test.db")  # First create the DB
    poolfile = CoreDB("test.db", ro_mode=True)  # Open in RO
    with pytest.raises(SQLAlchemyError):
        poolfile.add_trajectory("test.xml", "")
    Path("./test.db").unlink(missing_ok=True)


def test_add_traj_and_update_to_db():
    """Add and update a trajectory to CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    assert poolfile.fetch_trajectory(0)[0] == "test.xml"
    poolfile.update_trajectory(0, "UpdatedTest.xml", "")
    assert poolfile.fetch_trajectory(0)[0] == "UpdatedTest.xml"
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_try_update_traj_to_db():
    """Try update missing trajectory to CoreDB."""
    poolfile = CoreDB("test.db")
    with pytest.raises(ValueError):
        poolfile.update_trajectory(0, "UpdatedTest.xml", "dummy")
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_try_update_weight_to_db():
    """Try updating weight to missing trajectory to CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    with pytest.raises(ValueError):
        poolfile.update_trajectory_weight(3, 1.0)
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_add_traj_to_db_inmemory():
    """Add a trajectory to SQL database in memory."""
    poolfile = CoreDB("", in_memory=True)
    poolfile.add_trajectory("test.xml", "")
    assert poolfile.get_trajectory_count() == 1


@pytest.mark.usefixtures("skip_on_windows")
def test_add_traj_to_missing_db():
    """Add a trajectory to a deleted CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    assert poolfile.get_trajectory_count() == 1
    Path("./test.db").unlink(missing_ok=True)
    with pytest.raises(SQLAlchemyError):
        poolfile.add_trajectory("test2.xml", "")


def test_archive_traj_to_db():
    """Archive a trajectory to CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.archive_trajectory("test.xml", "")
    assert poolfile.get_archived_trajectory_count() == 1
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_add_traj_and_fetch():
    """Add a trajectory and fetch from CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    poolfile.add_trajectory("test_2.xml", "")
    assert poolfile.get_trajectory_count() == 2
    traj, _ = poolfile.fetch_trajectory(0)
    assert traj == "test.xml"
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_fetch_unknown_traj():
    """Fetch an unknown trajectory."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    assert poolfile.get_trajectory_count() == 1
    with pytest.raises(ValueError):
        _, _ = poolfile.fetch_trajectory(1)
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_archive_and_fetch_traj_to_db():
    """Archive a trajectory to CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.archive_trajectory("test.xml", "")
    assert poolfile.get_archived_trajectory_count() == 1
    traj, _ = poolfile.fetch_archived_trajectory(0)
    assert traj == "test.xml"
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_archive_and_delete_traj_to_db():
    """Archive then delete trajectory to CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.archive_trajectory("test.xml", "")
    assert poolfile.get_archived_trajectory_count() == 1
    poolfile.discard_archived_trajectory(0)
    assert poolfile.get_archived_trajectory_count() == 0
    del poolfile
    Path("./test.db").unlink(missing_ok=True)

def test_archive_and_clear_db():
    """Archive multiple traj then clear archive."""
    poolfile = CoreDB("test.db")
    poolfile.archive_trajectory("test_01.xml", "")
    poolfile.archive_trajectory("test_02.xml", "")
    assert poolfile.get_archived_trajectory_count() == 2
    poolfile.clear_archived_trajectories()
    assert poolfile.get_archived_trajectory_count() == 0
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_fetch_unknown_archived_traj():
    """Fetch an unknown archived trajectory."""
    poolfile = CoreDB("test.db")
    poolfile.archive_trajectory("test.xml", "")
    assert poolfile.get_archived_trajectory_count() == 1
    with pytest.raises(ValueError):
        _, _ = poolfile.fetch_archived_trajectory(1)
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_lock_trajectory():
    """Lock a trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    status = poolfile.lock_trajectory(0)
    assert status
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_lock_locked_trajectory():
    """Lock an already locked trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    status = poolfile.lock_trajectory(0)
    status = poolfile.lock_trajectory(0)
    assert status is False
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_lock_and_release_trajectory():
    """Lock and release a trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    status = poolfile.lock_trajectory(0)
    poolfile.release_trajectory(0)
    status = poolfile.lock_trajectory(0)
    assert status
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_lock_and_complete_trajectory():
    """Lock and mark complete a trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    _ = poolfile.lock_trajectory(0)
    poolfile.mark_trajectory_as_completed(0)
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_lock_and_complete_unknown_trajectory():
    """Lock and try to mark a trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    _ = poolfile.lock_trajectory(0)
    with pytest.raises(ValueError):
        poolfile.mark_trajectory_as_completed(1)
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_lock_and_release_multiple_trajectory():
    """Lock and release several trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    for i in range(10):
        poolfile.add_trajectory(f"test{i}.xml", "")
        status = poolfile.lock_trajectory(0)
    poolfile.release_all_trajectories()
    status = True
    for _ in range(10):
        status = status or poolfile.lock_trajectory(0)
    assert status
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_lock_unknown_trajectory():
    """Lock an unknown trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    with pytest.raises(ValueError):
        _ = poolfile.lock_trajectory(1)
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_update_chkfile_trajectory():
    """Update only the traj chkfile."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    poolfile.update_trajectory_file(0, "test_updated.xml")
    assert poolfile.fetch_trajectory(0)[0] == "test_updated.xml"
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_lock_in_missing_db():
    """Lock a trajectory in a missing CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    Path("./test.db").unlink(missing_ok=True)
    with pytest.raises(SQLAlchemyError):
        _ = poolfile.lock_trajectory(0)


def test_release_unknown_trajectory():
    """Release an unknown trajectory in the CoreDB."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    with pytest.raises(ValueError):
        poolfile.release_trajectory(1)
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_splitting_data_add():
    """Adding splitting data to the database."""
    poolfile = CoreDB("test.db")
    amsfile = AMSDB(poolfile.engine())
    for i in range(10):
        amsfile.add_splitting_data(i, 1, 0.1, [i - 1], [0], [0.0], [0.0, 0.0])
        amsfile.mark_last_iteration_as_completed()
    del amsfile
    del poolfile
    Path("./test.db").unlink(missing_ok=True)


def test_splitting_data_add_and_ongoing():
    """Adding splitting data to the database."""
    poolfile = CoreDB("", in_memory=True)
    amsfile = AMSDB(poolfile.engine())
    amsfile.mark_last_iteration_as_completed()
    for i in range(10):
        amsfile.add_splitting_data(i, 1, 0.1, [i - 1], [0], [0.0], [0.0, 0.0])
        amsfile.mark_last_iteration_as_completed()
    assert amsfile.get_ongoing() is None
    amsfile.add_splitting_data(10, 1, 0.1, [10 - 1, 1, 56], [0], [0.0], [0.0, 0.0])
    assert amsfile.get_ongoing() == [9, 1, 56]


def test_splitting_data_add_and_query():
    """Adding splitting data to the database."""
    poolfile = CoreDB("", in_memory=True)
    amsfile = AMSDB(poolfile.engine())
    for i in range(1, 2):
        amsfile.add_splitting_data(2 * i, 1, 0.1, [2 * i - 1], [0], [0.0], [0.0, 0.0])
        amsfile.mark_last_iteration_as_completed()
    assert np.all(amsfile.get_minmax()[0] == np.array([2.0, 0.0, 0.0], dtype="float64"))


def test_splitting_data_add_and_discard_last():
    """Adding splitting data to the database."""
    poolfile = CoreDB("", in_memory=True)
    amsfile = AMSDB(poolfile.engine())
    for i in range(1, 4):
        amsfile.add_splitting_data(2 * i, 1, 0.1, [2 * i - 1], [0], [0.0], [i*0.01, i*0.05])
        amsfile.mark_last_iteration_as_completed()
    assert np.all(np.isclose(amsfile.get_minmax()[-1], np.array([6.0, 0.03, 0.15], dtype="float64")))
    amsfile.discard_last_iteration()
    assert np.all(np.isclose(amsfile.get_minmax()[-1], np.array([4.0, 0.02, 0.1], dtype="float64")))

def test_splitting_data_add_dump_json():
    """Adding splitting data to the database."""
    poolfile = CoreDB("", in_memory=True)
    amsfile = AMSDB(poolfile.engine())
    for i in range(1, 4):
        amsfile.add_splitting_data(2 * i, 1, 0.1, [2 * i - 1], [0], [0.0], [i*0.01, i*0.05])
        amsfile.mark_last_iteration_as_completed()
    amsfile.dump_file_json("test.json")
    assert Path("./test.json").exists() is True
    Path("./test.json").unlink()


def test_splitting_data_add_update_and_query():
    """Adding splitting data to the database."""
    poolfile = CoreDB("", in_memory=True)
    amsfile = AMSDB(poolfile.engine())
    amsfile.add_splitting_data(2, 1, 0.1, [1], [0], [0.0], [0.0, 0.0])
    amsfile.update_splitting_data(2, 1, 0.1, [1], [0], [0.0], [0.0, 0.3])
    amsfile.mark_last_iteration_as_completed()
    assert np.all(amsfile.get_minmax()[0] == np.array([2.0, 0.0, 0.3], dtype="float64"))


@pytest.mark.usefixtures("skip_on_windows")
def test_splitting_data_query_fail():
    """Adding splitting data to the database."""
    poolfile = CoreDB("test.db")
    amsfile = AMSDB(poolfile.engine())
    for i in range(1):
        amsfile.add_splitting_data(2 * i, 1, 0.1, [2 * i - 1], [0], [0.0], [0.0, 0.0])
    assert amsfile.get_k_split() == 1

    poolfile = CoreDB("test.db", ro_mode=True)
    amsfile = AMSDB(poolfile.engine())
    with pytest.raises(SQLAlchemyError):
        amsfile.mark_last_iteration_as_completed()
    Path("./test.db").unlink(missing_ok=True)


def test_dump_json():
    """Dump the content of the DB to a json file."""
    poolfile = CoreDB("test.db")
    poolfile.add_trajectory("test.xml", "")
    poolfile.archive_trajectory("test_arch.xml", "")
    poolfile.dump_file_json()
    assert Path("./test.json").exists() is True
    del poolfile
    Path("./test.db").unlink(missing_ok=True)
    Path("./test.json").unlink(missing_ok=True)
