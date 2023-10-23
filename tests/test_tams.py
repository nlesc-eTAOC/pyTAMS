"""Tests for the pytams.tams class."""
from pytams.tams import TAMS
from tests.models import DoubleWellModel
from tests.models import SimpleFModel


def test_initTAMS():
    """Test TAMS initialization."""
    fmodel = {}
    parameters = {}
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    assert tams.nTraj == 500


def test_simpleModelTAMS():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel()
    parameters = {
        "nTrajcetories": 100,
        "nSplitIter": 200,
        "Verbose": True,
        "nProc": 1,
        "traj.end_time": 0.02,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.15,
    }
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0


def test_doublewellModelTAMS():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel()
    parameters = {
        "nTrajcetories": 100,
        "nSplitIter": 400,
        "Verbose": True,
        "nProc": 1,
        "traj.end_time": 10.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.8,
        "traj.stoichForcing": 2.0,
    }
    tams = TAMS(fmodel=fmodel, parameters=parameters)
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.005
