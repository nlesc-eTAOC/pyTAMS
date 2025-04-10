"""Tests for the pytams.trajectory class."""
from math import isclose
from pathlib import Path
import pytest
from pytams.fmodel import ForwardModelBaseClass
from pytams.trajectory import Snapshot
from pytams.trajectory import Trajectory
from pytams.utils import moving_avg
from tests.models import SimpleFModel


def test_initSnapshot():
    """Test initialization of a snapshot."""
    snap = Snapshot(0.1,0.1,"Noisy","State")
    assert snap.time == 0.1
    assert snap.has_state()


def test_initSnapshotNoState():
    """Test initialization of a stateless snapshot."""
    snap = Snapshot(0.1,0.1,"Noisy")
    assert not snap.has_state()


def test_initBaseClassError():
    """Test using base class fmodel during trajectory creation."""
    fmodel = ForwardModelBaseClass
    parameters = {}
    with pytest.raises(Exception):
        _ = Trajectory(1, fmodel, parameters)


def test_initBlankTraj():
    """Test blank trajectory creation."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 2.0,
                                  "step_size": 0.01}}
    t_test = Trajectory(1, fmodel, parameters)
    assert t_test.id() == 1
    assert t_test.idstr() == "traj000001"
    assert t_test.current_time() == 0.0
    assert t_test.score_max() == 0.0


def test_initParametrizedTraj():
    """Test parametrized trajectory creation."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 2.0,
                                  "step_size": 0.01,
                                  "targetscore": 0.25}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.set_work_dir(Path("."))
    assert t_test.step_size() == 0.01


def test_restartEmptyTraj():
    """Test (empty) trajectory restart."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 2.0,
                                  "step_size": 0.01}}
    fromTraj = Trajectory(1, fmodel, parameters)
    rstTraj = Trajectory(2, fmodel, parameters)
    rst_test = Trajectory.branch_from_trajectory(fromTraj, rstTraj, 0.1)
    assert rst_test.current_time() == 0.0


def test_simpleModelTraj():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.advance(0.01)
    assert isclose(t_test.score_max(), 0.1, abs_tol=1e-9)
    assert t_test.is_converged() is False
    t_test.advance()
    assert t_test.is_converged() is True


def test_storeAndRestoreSimpleTraj():
    """Test store and restoring trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.05,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.advance(0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkFile = Path("./test.xml")
    t_test.store(chkFile)
    assert chkFile.exists() is True
    rst_test = Trajectory.restore_from_checkfile(chkFile, fmodel, parameters)
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkFile.unlink()


def test_storeAndRestoreFrozenSimpleTraj():
    """Test store and restoring frozen trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.05,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.advance(0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkFile = Path("./test.xml")
    t_test.store(chkFile)
    assert chkFile.exists() is True
    rst_test = Trajectory.restore_from_checkfile(chkFile, fmodel, parameters, frozen=True)
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    with pytest.raises(Exception):
        rst_test.advance()
    with pytest.raises(Exception):
        rst_test.one_step()


def test_restartSimpleTraj():
    """Test trajectory restart."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    fromTraj = Trajectory(1, fmodel, parameters)
    fromTraj.advance(0.01)
    rstTraj = Trajectory(2, fmodel, parameters)
    rst_test = Trajectory.branch_from_trajectory(fromTraj, rstTraj, 0.05)
    assert rst_test.current_time() == 0.006


def test_accessDataSimpleTraj():
    """Test trajectory data access."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.advance(0.01)
    assert t_test.get_length() == 11
    assert isclose(t_test.get_time_array()[-1], 0.01, abs_tol=1e-9)
    assert isclose(t_test.get_score_array()[-1], 0.1, abs_tol=1e-9)

def test_sparseSimpleTraj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25,
                                  "sparse_freq": 5}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.advance(0.012)
    assert isclose(t_test.score_max(), 0.12, abs_tol=1e-9)
    assert t_test.is_converged() is False
    assert isclose(t_test.get_last_state(), 0.01, abs_tol=1e-9)
    t_test.advance()
    assert t_test.is_converged() is True
    assert isclose(t_test.get_last_state(), 0.025, abs_tol=1e-9)

def test_storeAndRestartSparseSimpleTraj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.04,
                                  "step_size": 0.001,
                                  "targetscore": 0.25,
                                  "sparse_freq": 5}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.advance(0.013)
    assert isclose(t_test.score_max(), 0.13, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkFile = Path("./test.xml")
    t_test.store(chkFile)
    assert chkFile.exists() is True
    rst_test = Trajectory.restore_from_checkfile(chkFile, fmodel, parameters)
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkFile.unlink()

def test_scoreMovingAverage():
    """Test using a moving average on a score array."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.9,
                                  "step_size": 0.0001,
                                  "targetscore": 0.95}}
    t_test = Trajectory(1, fmodel, parameters)
    t_test.advance()
    score = t_test.get_score_array()
    avg_score = moving_avg(score, 10)
    assert isclose(avg_score[0],0.0045,abs_tol=1e-9)
