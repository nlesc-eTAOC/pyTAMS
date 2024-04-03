"""Tests for the pytams.tams class."""
import os
import pytest
from pytams.fmodel import ForwardModel
from pytams.tams import TAMS
from tests.models import DoubleWellModel
from tests.models import SimpleFModel


def test_initTAMS():
    """Test TAMS initialization."""
    fmodel = ForwardModel
    parameters = {}
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    assert tams.nTraj() == 500


def test_simpleModelTAMS():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 200,
        "Verbose": False,
        "traj.end_time": 0.02,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.15,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0


def test_simpleModelTAMSSlurmFail():
    """Test TAMS with simple model with Slurm dask backend."""
    fmodel = SimpleFModel
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 200,
        "Verbose": False,
        "dask.backend" : "slurm",
        "dask.slurm_config_file" : "dummy.yaml",
        "traj.end_time": 0.02,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.15,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    with pytest.raises(Exception):
        tams.compute_probability()


def test_simpleModelTwiceTAMS():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 200,
        "Verbose": False,
        "DB_save": True,
        "DB_prefix": "simpleModelTest",
        "traj.end_time": 0.02,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.15,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0
    # Re-init TAMS and run to test competing database
    # on disk.
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    ndb = 0
    for folder in os.listdir("."):
        if "simpleModelTest" in str(folder):
            ndb += 1
    assert ndb == 2


def test_stallingSimpleModelTAMS():
    """Test TAMS with simple model and stalled score function."""
    fmodel = SimpleFModel
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 200,
        "Verbose": True,
        "traj.end_time": 1.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 1.1,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    with pytest.raises(Exception):
      tams.compute_probability()


def test_doublewellModelTAMS():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 400,
        "Verbose": True,
        "wallTime": 500.0,
        "traj.end_time": 10.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.8,
        "traj.stoichForcing": 0.8,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2


def test_doublewellModel2WorkersTAMS():
    """Test TAMS with the doublewell model using two workers."""
    fmodel = DoubleWellModel
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 400,
        "Verbose": True,
        "dask.nworker_init": 2,
        "dask.nworker_iter": 2,
        "DB_save": True,
        "DB_prefix": "dwTest",
        "wallTime": 500.0,
        "traj.end_time": 10.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.8,
        "traj.stoichForcing": 0.8,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2


def test_doublewellModel2WorkersRestoreTAMS():
    """Test TAMS with the doublewell model using two workers and restoring."""
    fmodel = DoubleWellModel
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 400,
        "Verbose": True,
        "dask.nworker_init": 2,
        "dask.nworker_iter": 2,
        "DB_save": True,
        "DB_prefix": "dwTest",
        "DB_restart": "dwTest.tdb",
        "wallTime": 500.0,
        "traj.end_time": 10.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.8,
        "traj.stoichForcing": 0.8,
    }
    tams = TAMS(fmodel_t=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2
