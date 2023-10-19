"""Tests for the pytams.trajectory class."""
from pytams.trajectory import Trajectory


def test_initBlankTraj():
    """Test blank trajectory creation."""
    fmodel = {}
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    assert t_test.id() == "ttest"
    assert t_test.ctime() == 0.0
    assert t_test.scoreMax() == 0.0


def test_initParametrizedTraj():
    """Test paramtrized trajectory creation."""
    fmodel = {}
    parameters = {
        "traj.end_time": 2.0,
        "traj.step_size": 0.01,
        "traj.targetScore": 0.25,
    }
    t_test = Trajectory(fmodel, parameters)
    assert t_test.stepSize() == 0.01


def test_restartTraj():
    """Test (empty) trajectory restart."""
    fmodel = {}
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    rst_test = Trajectory.restartFromTraj(t_test, 0.1)
    assert rst_test.ctime() == 0.0
