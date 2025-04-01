"""Tests for the pytams.tams class."""
import os
import shutil
import pytest
import toml
from pytams.tams import TAMS
from pytams.tams import TAMSError
from tests.models import DoubleWellModel
from tests.models import SimpleFModel


def test_initTAMS():
    """Test TAMS initialization."""
    fmodel = SimpleFModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 500, "nsplititer": 200},
                   "trajectory" : {"end_time": 0.02, "step_size": 0.001}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    assert tams.nTraj() == 500

def test_initTAMSMissingReq():
    """Test failed TAMS initialization."""
    fmodel = SimpleFModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"nsplititer": 200},
                   "trajectory" : {"end_time": 0.02, "step_size": 0.001}}, f)
    with pytest.raises(ValueError):
        _ = TAMS(fmodel_t=fmodel, a_args=[])

def test_initTAMSNoInput():
    """Test failed TAMS initialization."""
    fmodel = SimpleFModel
    with pytest.raises(TAMSError):
        _ = TAMS(fmodel_t=fmodel, a_args=["-i", "dummy.toml"])

def test_simpleModelTAMS():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING"},
                   "runner": {"type" : "asyncio"},
                   "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0


def test_simpleModelTAMSwithDB():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING"},
                   "runner": {"type" : "dask"},
                   "database" : {"path" : "simpleModelTest.tdb"},
                   "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15,
                                  "chkfile_dump_all" : True}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0
    shutil.rmtree("simpleModelTest.tdb")

def test_simpleModelTAMSSlurmFail():
    """Test TAMS with simple model with Slurm dask backend."""
    fmodel = SimpleFModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "DEBUG"},
                   "runner": {"type" : "dask"},
                   "dask": {"backend" : "slurm", "slurm_config_file": "dummy.yaml"},
                   "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    with pytest.raises(Exception):
        tams.compute_probability()


def test_simpleModelTwiceTAMS():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING",
                            "logfile": "test.log"},
                   "runner": {"type" : "asyncio"},
                   "database" : {"path" : "simpleModelTest.tdb", "restart" : True},
                   "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0
    # Re-init TAMS and run to test competing database
    # on disk.
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    ndb = 0
    for folder in os.listdir("."):
        if "simpleModelTest" in str(folder):
            shutil.rmtree(folder)
            ndb += 1
    assert ndb == 2
    assert os.path.exists("test.log")
    os.remove("test.log")


def test_stallingSimpleModelTAMS():
    """Test TAMS with simple model and stalled score function."""
    fmodel = SimpleFModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "ERROR"},
                   "runner": {"type" : "asyncio"},
                   "trajectory": {"end_time": 1.0, "step_size": 0.01, "targetscore": 1.1}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    with pytest.raises(Exception):
      tams.compute_probability()


def test_doublewellModelTAMS():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 400, "walltime": 500.0},
                   "runner": {"type" : "dask"},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.8, "stoichforcing" : 0.8}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2


def test_doublewellModelSaveTAMS():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 50, "nsplititer": 100, "walltime": 500.0},
                   "runner": {"type" : "dask"},
                   "database": {"path": "dwTest.tdb"},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.3, "stoichforcing" : 0.8}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2
    os.remove("input.toml")
    shutil.rmtree("dwTest.tdb")


def test_doublewellDeterministicModelTAMS():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 400,
                            "walltime": 500.0, "deterministic": True},
                   "runner": {"type" : "asyncio"},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.8, "stoichforcing" : 0.8}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 0.39663733322475747
    os.remove("input.toml")


@pytest.mark.dependency()
def test_doublewellModel2WorkersTAMS():
    """Test TAMS with the doublewell model using two workers."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 400,
                            "walltime": 500.0, "deterministic": True},
                   "runner": {"type" : "dask", "nworker_init": 2, "nworker_iter": 2},
                   "database": {"path": "dwTest.tdb"},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.6, "stoichforcing" : 0.8}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 0.5238831403348925
    os.remove("input.toml")


@pytest.mark.dependency(depends=["test_doublewellModel2WorkersTAMS"])
def test_doublewellModel2WorkersRestoreTAMS():
    """Test TAMS with the doublewell model using two workers and restoring."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 100, "nsplititer": 400, "walltime": 500.0},
                   "database": {"path": "dwTest.tdb"},
                   "runner": {"type" : "asyncio", "nworker_init": 2, "nworker_iter": 2},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.6, "stoichforcing" : 0.8}}, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2
    os.remove("input.toml")
    shutil.rmtree("dwTest.tdb")


def test_doublewellVerySlowTAMS():
    """Test TAMS run out of time with a slow doublewell."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 5, "nsplititer": 400, "walltime": 6.0},
                   "database": {"path": "dwTest.tdb"},
                   "runner": {"type" : "dask", "nworker_init": 1, "nworker_iter": 1},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.7, "stoichforcing" : 0.1},
                   "model": {"slow_factor": 0.2}}
                  , f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba <= 0.0
    os.remove("input.toml")
    shutil.rmtree("dwTest.tdb")

@pytest.mark.dependency()
def test_doublewellSlowTAMS():
    """Test TAMS run out of time with a slow doublewell."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 5, "nsplititer": 400, "walltime": 6.0},
                   "database": {"path": "dwTest.tdb"},
                   "runner": {"type" : "asyncio", "nworker_init": 1, "nworker_iter": 1},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.7, "stoichforcing" : 0.1},
                   "model": {"slow_factor": 0.003}}
                  , f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba <= 0.0
    os.remove("input.toml")

@pytest.mark.dependency(depends=["test_doublewellSlowTAMS"])
def test_doublewellSlowRestoreTAMS():
    """Test TAMS restarting a slow doublewell."""
    fmodel = DoubleWellModel
    with open("input.toml", 'w') as f:
        toml.dump({"tams": {"ntrajectories": 5, "nsplititer": 400, "walltime": 6.0},
                   "database": {"path": "dwTest.tdb"},
                   "runner": {"type" : "asyncio", "nworker_init": 1, "nworker_iter": 1},
                   "trajectory": {"end_time": 10.0, "step_size": 0.01,
                                  "targetscore": 0.7, "stoichforcing" : 0.1},
                   "model": {"slow_factor": 0.003}}
                  , f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba <= 0.0
    os.remove("input.toml")
    shutil.rmtree("dwTest.tdb")
