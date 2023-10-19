"""Tests for the pytams.trajectory class."""
from pytams.trajectory import Trajectory


def test_initTraj():
    """Test trajectory creation."""
    fmodel = {}
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    assert t_test.id() == "ttest"
