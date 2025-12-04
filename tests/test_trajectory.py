"""Tests for the pytams.trajectory class."""

from math import isclose
from pathlib import Path
import pytest
from pytams.fmodel import ForwardModelBaseClass
from pytams.trajectory import Snapshot
from pytams.trajectory import Trajectory
from pytams.utils import moving_avg
from tests.models import DoubleWellModel
from tests.models import SimpleFModel


def test_init_snapshot():
    """Test initialization of a snapshot."""
    snap = Snapshot(0.1, 0.1, "Noisy", "State")
    assert snap.time == 0.1
    assert snap.has_state()


def test_init_snapshot_nostate():
    """Test initialization of a stateless snapshot."""
    snap = Snapshot(0.1, 0.1, "Noisy")
    assert not snap.has_state()


def test_init_missing_basic_inputs():
    """Test lack of minimal traj data in trajectory creation."""
    fmodel = ForwardModelBaseClass
    parameters = {}
    with pytest.raises(ValueError):
        _ = Trajectory(1, 1.0, fmodel, parameters)


def test_init_baseclasserror():
    """Test using base class fmodel during trajectory creation."""
    fmodel = ForwardModelBaseClass
    parameters = {"trajectory": {"end_time": 2.0, "step_size": 0.01}}
    with pytest.raises(TypeError):
        _ = Trajectory(1, 1.0, fmodel, parameters)


def test_init_blank_traj():
    """Test blank trajectory creation."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 2.0, "step_size": 0.01}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    assert t_test.id() == 1
    assert t_test.idstr() == "traj000001_0000"
    assert t_test.current_time() == 0.0
    assert t_test.score_max() == -1000000000000.0


def test_init_parametrized_traj():
    """Test parametrized trajectory creation."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 2.0, "step_size": 0.01, "targetscore": 0.25}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.set_workdir(Path())
    assert t_test.step_size() == 0.01


def test_restart_empty_traj():
    """Test (empty) trajectory restart."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 2.0, "step_size": 0.01}}
    from_traj = Trajectory(1, 0.5, fmodel, parameters)
    rst_traj = Trajectory(2, 0.5, fmodel, parameters)
    rst_test = Trajectory.branch_from_trajectory(from_traj, rst_traj, 0.1, 0.25)
    assert rst_test.current_time() == 0.0


def test_simple_model_traj():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance(0.01)
    assert isclose(t_test.score_max(), 0.1, abs_tol=1e-9)
    assert t_test.is_converged() is False
    t_test.advance()
    assert t_test.is_converged() is True


def test_branch_simple_model_traj():
    """Test branching a trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.04, "step_size": 0.0002, "targetscore": 0.45}}
    t_ancestor = Trajectory(1 ,0.5, fmodel, parameters)
    t_ancestor.advance()
    assert t_ancestor.get_computed_steps_count() == 201
    t_branched = Trajectory(2, 0.5, fmodel, parameters)
    t_branched = Trajectory.branch_from_trajectory(t_ancestor, t_branched, 0.1, 0.25)
    assert t_branched.get_computed_steps_count() == 0
    t_branched.advance()
    assert t_branched.get_computed_steps_count() == 150


def test_store_and_restore_simple_traj():
    """Test store and restoring trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.05, "step_size": 0.001, "targetscore": 0.25}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance(0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.serialize_metadata_json()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, parameters)
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkfile.unlink(missing_ok=True)


def test_store_and_restore_frozen_simple_traj():
    """Test store and restoring frozen trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.05, "step_size": 0.001, "targetscore": 0.25}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance(0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.serialize_metadata_json()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, parameters, frozen=True)
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    with pytest.raises(RuntimeError):
        rst_test.advance()
    with pytest.raises(RuntimeError):
        rst_test._one_step()
    chkfile.unlink(missing_ok=True)


def test_restart_simple_traj():
    """Test trajectory restart."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}}
    from_traj = Trajectory(1, 0.5, fmodel, parameters)
    from_traj.advance(0.01)
    rst_traj = Trajectory(2, 0.5, fmodel, parameters)
    rst_test = Trajectory.branch_from_trajectory(from_traj, rst_traj, 0.05, 0.25)
    assert rst_test.current_time() == 0.006


def test_access_data_simple_traj():
    """Test trajectory data access."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance(0.01)
    assert t_test.get_length() == 11
    assert isclose(t_test.get_time_array()[-1], 0.01, abs_tol=1e-9)
    assert isclose(t_test.get_score_array()[-1], 0.1, abs_tol=1e-9)


def test_sparse_simple_traj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25, "sparse_freq": 5}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance(0.012)
    assert isclose(t_test.score_max(), 0.12, abs_tol=1e-9)
    assert t_test.is_converged() is False
    assert isclose(t_test.get_last_state(), 0.009, abs_tol=1e-9)
    t_test.advance()
    assert t_test.is_converged() is True
    assert isclose(t_test.get_last_state(), 0.024, abs_tol=1e-9)


def test_sparse_simple_traj_access_states():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.04, "step_size": 0.0002, "targetscore": 0.25, "sparse_freq": 5}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance()
    assert len(t_test.get_state_list()) == 26


def test_store_and_restart_sparse_simple_traj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25, "sparse_freq": 5}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance(0.013)
    assert isclose(t_test.score_max(), 0.13, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.serialize_metadata_json()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, parameters)
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkfile.unlink()


def test_score_moving_average():
    """Test using a moving average on a score array."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.9, "step_size": 0.0001, "targetscore": 0.95}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance()
    score = t_test.get_score_array()
    avg_score = moving_avg(score, 10)
    assert isclose(avg_score[0], 0.0045, abs_tol=1e-9)


def test_sparse_dw_traj_with_restore():
    """Test restore a sparse trajectory with DW model."""
    fmodel = DoubleWellModel
    parameters = {
        "tams": {"deterministic": True},
        "trajectory": {"end_time": 15.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10},
        "model": {"noise_amplitude": 0.8},
    }
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    t_test.advance(4.07)
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert isclose(t_test.score_max(), 0.5383998247480907, abs_tol=1e-9)
    assert not t_test.is_converged()
    metadata = t_test.serialize_metadata_json()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, parameters, frozen=False)
    rst_test.advance()
    assert rst_test.score_max() > 0.95
    assert rst_test.is_converged()
    chkfile.unlink(missing_ok=True)


def test_sparse_dw_traj_with_branching():
    """Test branching a sparse trajectory with simple model."""
    fmodel = DoubleWellModel
    parameters = {
        "tams": {"deterministic": True},
        "trajectory": {"end_time": 2.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10},
        "model": {"noise_amplitude": 0.3},
    }
    t_test = [Trajectory(1, 0.5, fmodel, parameters), Trajectory(2, 0.5, fmodel, parameters)]
    t_test[0].advance()
    t_test[1].advance()
    if t_test[0].score_max() > t_test[1].score_max():
        rst_idx = 1
        from_idx = 0
        rst_val = t_test[1].score_max()
    else:
        rst_idx = 0
        from_idx = 1
        rst_val = t_test[0].score_max()
    branched_test = Trajectory.branch_from_trajectory(t_test[from_idx], t_test[rst_idx], rst_val, 0.25)
    branched_test.advance()
    assert branched_test.score_max() > t_test[rst_idx].score_max()
