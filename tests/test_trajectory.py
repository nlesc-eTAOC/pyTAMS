"""Tests for the pytams.trajectory class."""
import os
from math import isclose
import pytest
from pytams.fmodel import ForwardModel
from pytams.trajectory import Snapshot, Trajectory
from tests.models import SimpleFModel


def test_initSnapshot():
    """Test initialization of a snapshot."""
    snap = Snapshot(0.1,0.1,"Noisy","State")
    assert snap.Time() == 0.1
    assert snap.hasState()


def test_initSnapshotNoState():
    """Test initialization of a stateless snapshot."""
    snap = Snapshot(0.1,0.1,"Noisy")
    assert not snap.hasState()


def test_initBlankTraj():
    """Test blank trajectory creation."""
    fmodel = ForwardModel
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    assert t_test.id() == "ttest"
    assert t_test.ctime() == 0.0
    assert t_test.scoreMax() == 0.0


def test_initParametrizedTraj():
    """Test parametrized trajectory creation."""
    fmodel = ForwardModel
    parameters = {"trajectory" : {"end_time": 2.0,
                                  "step_size": 0.01,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, "ttest")
    assert t_test.stepSize() == 0.01


def test_restartEmptyTraj():
    """Test (empty) trajectory restart."""
    fmodel = ForwardModel
    parameters = {}
    t_test = Trajectory(fmodel, parameters, "ttest")
    rst_test = Trajectory.restartFromTraj(t_test, "ttest", 0.1)
    assert rst_test.ctime() == 0.0


def test_templateModelExceptions():
    """Test trajectory exception with template model."""
    fmodel = ForwardModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, "ttest")
    with pytest.raises(Exception):
        t_test.advance()


def test_simpleModelTraj():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, "Traj1")
    t_test.advance(0.01)
    assert isclose(t_test.scoreMax(), 0.1, abs_tol=1e-9)
    assert t_test.isConverged() is False
    t_test.advance()
    assert t_test.isConverged() is True


def test_storeAndRestoreSimpleTraj():
    """Test store and restoring trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.05,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, "Traj1")
    t_test.advance(0.02)
    assert isclose(t_test.scoreMax(), 0.2, abs_tol=1e-9)
    assert t_test.isConverged() is False
    t_test.store("test.xml")
    assert os.path.exists("test.xml") is True
    rst_test = Trajectory.restoreFromChk("test.xml", fmodel, parameters)
    assert isclose(rst_test.scoreMax(), 0.2, abs_tol=1e-9)
    rst_test.advance()
    assert rst_test.isConverged() is True


def test_restartSimpleTraj():
    """Test trajectory restart."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, "rstSimple")
    t_test.advance(0.01)
    rst_test = Trajectory.restartFromTraj(t_test, "testiD", 0.05)
    assert rst_test.ctime() == 0.005


def test_accessDataSimpleTraj():
    """Test trajectory data access."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, "Simple")
    t_test.advance(0.01)
    assert t_test.getLength() == 11
    assert isclose(t_test.getTimeArr()[-1], 0.01, abs_tol=1e-9)
    assert isclose(t_test.getScoreArr()[-1], 0.1, abs_tol=1e-9)

def test_sparseSimpleTraj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25,
                                  "sparse_int": 5}}
    t_test = Trajectory(fmodel, parameters, "Traj1")
    t_test.advance(0.012)
    assert isclose(t_test.scoreMax(), 0.12, abs_tol=1e-9)
    assert t_test.isConverged() is False
    assert isclose(t_test.getLastState(), 0.01, abs_tol=1e-9)
    t_test.advance()
    assert t_test.isConverged() is True
    assert isclose(t_test.getLastState(), 0.025, abs_tol=1e-9)

def test_storeAndRestartSparseSimpleTraj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25,
                                  "sparse_int": 5}}
    t_test = Trajectory(fmodel, parameters, "Traj1")
    t_test.advance(0.013)
    assert isclose(t_test.scoreMax(), 0.13, abs_tol=1e-9)
    assert t_test.isConverged() is False
    t_test.store("test.xml")
    assert os.path.exists("test.xml") is True
    rst_test = Trajectory.restoreFromChk("test.xml", fmodel, parameters)
    rst_test.advance()
    assert rst_test.isConverged() is True
