"""Tests for the pytams.database class."""
import os
import shutil
import toml
from pytams.database import Database
from pytams.tams import TAMS
from tests.models import DoubleWellModel


def test_generateAndLoadTDB():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 400, "walltime": 500.0},
                   "database": {"DB_save": True, "DB_prefix": "dwTest"},
                   "dask": {"nworker_init": 2, "nworker_iter": 2},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.7, "stoichforcing" : 0.8}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    tams.compute_probability()

    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb

def test_exploreTDB():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"DB_restart": "dwTest.tdb"}}

    tdb = Database(fmodel, params_load_db)
    tdb.info()
    tdb.plotScoreFunctions("test.png")
    assert tdb.countEndedTraj() == 100
    assert tdb.getTransitionProbability() > 0.2
    shutil.rmtree("dwTest.tdb")
    os.remove("test.png")
