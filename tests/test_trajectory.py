"""Tests for the pytams.trajectory class."""
import pytest
from pytams.trajectory import Trajectory

def test_initTraj():
    fmodel = {}
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    assert(t_test.id() == "ttest")
