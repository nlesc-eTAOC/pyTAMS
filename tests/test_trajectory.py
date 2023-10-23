"""Tests for the pytams.trajectory class."""
import os
from math import isclose
import pytest
from pytams.fmodel import ForwardModel
from pytams.trajectory import Trajectory


class SimpleFModel(ForwardModel):
    """Simple forward model.

    The state is the time and score
    10 times the state, ceiled to 1.0
    """

    def __init__(self):
        """Override the template."""
        self._state = 0.0

    def advance(self, dt: float, forcingAmpl: float):
        """Override the template."""
        self._state = self._state + dt

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state):
        """Override the template."""
        self._state = state

    def score(self):
        """Override the template."""
        return min(self._state * 10.0, 1.0)


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
    t_test.store("test.xml")
    assert os.path.exists("test.xml") is True


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
