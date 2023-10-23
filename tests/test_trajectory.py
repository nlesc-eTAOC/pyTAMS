"""Tests for the pytams.trajectory class."""
import os
from math import isclose
import pytest
from pytams.fmodel import ForwardModel
from pytams.trajectory import Trajectory
from tests.models import SimpleFModel


def test_initBlankTraj():
    """Test blank trajectory creation."""
    fmodel = {}
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    assert t_test.id() == "ttest"
    assert t_test.ctime() == 0.0
    assert t_test.scoreMax() == 0.0


def test_initParametrizedTraj():
    """Test parametrized trajectory creation."""
    fmodel = {}
    parameters = {
        "traj.end_time": 2.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.25,
    }
    t_test = Trajectory(fmodel, parameters)
    assert t_test.stepSize() == 0.01


def test_restartEmptyTraj():
    """Test (empty) trajectory restart."""
    fmodel = {}
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    rst_test = Trajectory.restartFromTraj(t_test, 0.1)
    assert rst_test.ctime() == 0.0


def test_templateModelExceptions():
    """Test trajectory exception with template model."""
    fmodel = ForwardModel()
    parameters = {
        "traj.end_time": 0.04,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.25,
    }
    t_test = Trajectory(fmodel, parameters)
    with pytest.raises(Exception):
        t_test.advance()


def test_simpleModelTraj():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel()
    parameters = {
        "traj.end_time": 0.04,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.25,
    }
    t_test = Trajectory(fmodel, parameters, "Traj1")
    t_test.advance(0.01)
    assert isclose(t_test.scoreMax(), 0.1, abs_tol=1e-9)
    assert t_test.isConverged() is False
    t_test.advance()
    assert t_test.isConverged() is True


def test_storeAndRestoreSimpleTraj():
    """Test store and restoring trajectory with simple model."""
    fmodel = SimpleFModel()
    parameters = {
        "traj.end_time": 0.05,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.25,
    }
    t_test = Trajectory(fmodel, parameters, "Traj1")
    t_test.advance(0.02)
    assert isclose(t_test.scoreMax(), 0.2, abs_tol=1e-9)
    assert t_test.isConverged() is False
    t_test.store("test.xml")
    assert os.path.exists("test.xml") is True
    rst_test = Trajectory.restoreFromChk("test.xml", fmodel)
    assert isclose(rst_test.scoreMax(), 0.2, abs_tol=1e-9)
    rst_test.advance()
    assert rst_test.isConverged() is True


def test_restartSimpleTraj():
    """Test trajectory restart."""
    fmodel = SimpleFModel()
    parameters = {
        "traj.end_time": 0.04,
        "traj.step_size": 0.001,
        "traj.targetScore": 0.25,
    }
    t_test = Trajectory(fmodel, parameters)
    t_test.advance(0.01)
    rst_test = Trajectory.restartFromTraj(t_test, 0.05)
    assert rst_test.ctime() == 0.005
