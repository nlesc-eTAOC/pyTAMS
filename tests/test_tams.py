"""Tests for the pytams.tams class."""
from pytams.tams import TAMS
from tests.models import DoubleWellModel
from tests.models import SimpleFModel


def test_initTAMS():
    """Test TAMS initialization."""
    fmodel = {}
    parameters = {}
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    assert tams.nTraj() == 500


def test_simpleModelTAMS():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel()
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 200,
        "Verbose": False,
        "nProc": 1,
        "traj.end_time": 0.02,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.15,
    }
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0

def test_simpleModelTwiceTAMS():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel()
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 200,
        "Verbose": False,
        "DB_save": True,
        "DB_prefix": "simpleModelTest",
        "nProc": 1,
        "traj.end_time": 0.02,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.15,
    }
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()


def test_stallingSimpleModelTAMS():
    """Test TAMS with simple model and stalled score function."""
    fmodel = SimpleFModel()
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 200,
        "Verbose": True,
        "nProc": 1,
        "traj.end_time": 1.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 1.1,
    }
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba == 0.0


def test_doublewellModelTAMS():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel()
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 400,
        "Verbose": True,
        "nProc": 1,
        "DB_save": True,
        "DB_prefix": "dwTest",
        "wallTime": 500.0,
        "traj.end_time": 10.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.8,
        "traj.stoichForcing": 0.8,
    }
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2

def test_doublewellModel2WorkersTAMS():
    """Test TAMS with the doublewell model using two workers."""
    fmodel = DoubleWellModel()
    parameters = {
        "nTrajectories": 100,
        "nSplitIter": 400,
        "Verbose": True,
        "nProc": 2,
        "DB_save": True,
        "DB_prefix": "dwTest",
        "wallTime": 500.0,
        "traj.end_time": 10.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.8,
        "traj.stoichForcing": 0.8,
    }
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2
