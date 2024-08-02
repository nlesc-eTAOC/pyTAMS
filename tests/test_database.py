"""Tests for the pytams.database class."""
import os
import shutil
import pytest
import toml
from pytams.database import Database
from pytams.tams import TAMS
from tests.models import DoubleWellModel


def test_failedDBInitNoNTraj():
    """Test init of TDB from scratch."""
    fmodel = DoubleWellModel
    params_load_db = {}
    with pytest.raises(Exception):
        Database(fmodel, params_load_db)


def test_failedDBInitNoSplit():
    """Test init of TDB from scratch."""
    fmodel = DoubleWellModel
    params_load_db = {}
    with pytest.raises(Exception):
        Database(fmodel, params_load_db, ntraj=10)


def test_wrongFormat():
    """Test database with unsupported format."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_format": "WRONG"}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    tdb._readHeader()
    with pytest.raises(Exception):
        tdb._writeMetadata()


def test_emptyDB():
    """Test database access on empty database."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_format": "WRONG"}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    assert tdb.getTransitionProbability() == 0.0


def test_generateAndLoadTDB():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 400, "walltime": 500.0},
                   "database": {"DB_save": True, "DB_prefix": "dwTest"},
                   "runner": {"type": "asyncio", "nworker_init": 2, "nworker_iter": 2},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.6, "stoichforcing" : 0.8}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    tams.compute_probability()

    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb

def test_accessPoolLength():
    """Test accessing database trajectory pool length."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb.isEmpty() is False


def test_accessEndedCount():
    """Test accessing database trajectory metadata."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb.countEndedTraj() == 100


def test_accessConvergedCount():
    """Test accessing database trajectory metadata."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb.countConvergedTraj() == 100


def test_replaceTrajInDB():
    """Test replacing a trajectory in the database."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)

    traj_zero = tdb.getTraj(0)
    tdb.overwriteTraj(1,traj_zero)
    assert tdb.getTraj(1).idstr() == "traj000000"


def test_accessTrajDataInDB():
    """Test accessing a trajectory in the database."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)

    traj = tdb.getTraj(0)
    times = traj.getTimeArr()
    scores = traj.getScoreArr()
    noises = traj.getNoiseArr()
    assert times.size > 0
    assert scores.size > 0
    assert noises.size > 0

def test_exploreTDB():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}

    tdb = Database(fmodel, params_load_db)
    tdb.info()
    tdb.plotScoreFunctions("test.png")
    assert tdb.getTransitionProbability() > 0.2
    shutil.rmtree("dwTest.tdb")
    os.remove("test.png")
