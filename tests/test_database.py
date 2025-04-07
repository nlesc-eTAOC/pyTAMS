"""Tests for the pytams.database class."""
import os
import shutil
from pathlib import Path
import pytest
import toml
from pytams.database import Database
from pytams.tams import TAMS
from tests.models import DoubleWellModel


def test_failedDBInitNoNTraj():
    """Test init of TDB from scratch missing argument."""
    fmodel = DoubleWellModel
    params_load_db = {}
    with pytest.raises(Exception):
        Database(fmodel, params_load_db)


def test_failedDBInitNoSplit():
    """Test init of TDB failing missing nsplit."""
    fmodel = DoubleWellModel
    params_load_db = {}
    with pytest.raises(Exception):
        Database(fmodel, params_load_db, ntraj=10)


def test_wrongFormat():
    """Test init of TDB with unsupported format."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb", "format": "WRONG"}}
    with pytest.raises(Exception):
        _ = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)


def test_initEmptyDBInMemory():
    """Test init database."""
    fmodel = DoubleWellModel
    params_load_db = {}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    assert tdb.name() == "TAMS_DoubleWellModel"


def test_initEmptyDBOnDisk():
    """Test init database on disk."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    assert tdb.name() == "dwTest.tdb"
    shutil.rmtree("dwTest.tdb")

def test_reinitEmptyDBOnDisk():
    """Test init database on disk."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTestDouble.tdb"}}
    _ = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    params_load_db = {"database": {"path": "dwTestDouble.tdb", "restart": True}}
    _ = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    ndb = 0
    for folder in os.listdir("."):
        if "dwTestDouble" in str(folder):
            shutil.rmtree(folder)
            ndb += 1
    assert ndb == 2

def test_initandLoadEmptyDBOnDisk():
    """Test init database on disk."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    assert tdb.name() == "dwTest.tdb"
    _ = Database.load(Path(tdb.path()))
    shutil.rmtree("dwTest.tdb")

@pytest.mark.dependency(name="genDB")
def test_generateAndLoadTDB():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 50, "nsplititer": 200, "walltime": 500.0, "loglevel": "DEBUG"},
                   "database": {"path": "dwTest.tdb"},
                   "runner": {"type": "asyncio", "nworker_init": 2, "nworker_iter": 2},
                   "model": {"noise_amplitude" : 0.8},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.2}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    tams.compute_probability()

    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb
    os.remove("input.toml")

@pytest.mark.dependency(depends=["genDB"])
def test_accessPoolLength():
    """Test accessing database trajectory pool length."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()
    assert tdb.is_empty() is False


@pytest.mark.dependency(depends=["genDB"])
def test_accessEndedCount():
    """Test accessing database trajectory metadata."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()
    assert tdb.count_ended_traj() == 50


@pytest.mark.dependency(depends=["genDB"])
def test_accessConvergedCount():
    """Test accessing database trajectory metadata."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()
    assert tdb.count_converged_traj() == 50


@pytest.mark.dependency(depends=["genDB"])
def test_copyAndAccess():
    """Test copying the database and accessing it."""
    shutil.copytree("dwTest.tdb", "dwTestCopy.tdb")
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTestCopy.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()
    assert tdb.count_converged_traj() == 50
    shutil.rmtree("dwTestCopy.tdb")


@pytest.mark.dependency(depends=["genDB"])
def test_replaceTrajInDB():
    """Test replacing a trajectory in the database."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()

    traj_zero = tdb.get_traj(0)
    tdb.overwrite_traj(1,traj_zero)
    assert tdb.get_traj(1).idstr() == "traj000000"


@pytest.mark.dependency(depends=["genDB"])
def test_accessTrajDataInDB():
    """Test accessing a trajectory in the database."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()

    traj = tdb.get_traj(0)
    times = traj.get_time_array()
    scores = traj.get_score_array()
    noises = traj.get_noise_array()
    assert times.size > 0
    assert scores.size > 0
    assert noises.size > 0

@pytest.mark.dependency(depends=["genDB"])
def test_exploreTDB():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()
    tdb.info()
    tdb.plot_score_functions("test.png")
    assert tdb.get_transition_probability() > 0.2
    shutil.rmtree("dwTest.tdb")
    os.remove("test.png")
