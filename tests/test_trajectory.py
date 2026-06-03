"""Tests for the pyrevs.trajectory class."""

from dataclasses import FrozenInstanceError
from math import isclose
from pathlib import Path
import pytest
from pyrevs.core import Config
from pyrevs.core import ForwardModelBaseClass
from pyrevs.core import Snapshot
from pyrevs.diagnostics import DiagnosticAnalyst
from pyrevs.trajectory import Trajectory
from pyrevs.trajectory import TrajectoryConfig
from pyrevs.utils.utils import moving_avg
from tests.dwmodel import DoubleWellModel
from tests.models import SimpleFModel


def test_init_snapshot():
    """Test initialization of a snapshot."""
    snap = Snapshot(time=0.1, score=0.1, noise="Noisy", state="State")
    assert snap.time == 0.1
    assert snap.has_state


def test_init_snapshot_nostate():
    """Test initialization of a stateless snapshot."""
    snap = Snapshot(time=0.1, score=0.1, noise="Noisy")
    assert not snap.has_state


def test_init_snapshot_negtime():
    """Test initialization with a negative time."""
    with pytest.raises(ValueError):
        _ = Snapshot(time=-0.1, score=0.1, noise="Noisy")


def test_modifying_snap():
    """Test modifying a snapshot."""
    snap = Snapshot(time=0.1, score=0.1, noise="Noisy", state="State")
    with pytest.raises(FrozenInstanceError):
        snap.state = "OtherState"


def test_init_missing_basic_inputs():
    """Test lack of minimal params in TrajectoryConfig."""
    parameters = {}
    cfg = Config(parameters)
    with pytest.raises(ValueError):
        cfg.load(TrajectoryConfig).validate()


def test_init_baseclasserror():
    """Test using base class fmodel during trajectory creation."""
    fmodel = ForwardModelBaseClass
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01}})
    with pytest.raises(TypeError):
        _ = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))


def test_init_blank_traj():
    """Test blank trajectory creation."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    assert t_test.id() == 1
    assert t_test.idstr() == "traj000001_0000"
    assert t_test.current_time() == 0.0
    assert t_test.score_max() == -1000000000000.0


def test_init_parametrized_traj():
    """Test parametrized trajectory creation."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.set_workdir(Path())
    assert t_test.step_size() == 0.01


def test_restart_empty_traj():
    """Test (empty) trajectory restart."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 2.0, "step_size": 0.01}})
    from_traj = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    rst_traj = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig))
    rst_test = Trajectory.branch_from_trajectory(from_traj, rst_traj, 0.1, 0.25)
    assert rst_test.current_time() == 0.0


def test_simple_model_traj():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.01)
    assert isclose(t_test.score_max(), 0.1, abs_tol=1e-9)
    assert t_test.is_converged() is False
    t_test.advance()
    assert t_test.is_converged() is True


def test_simple_model_traj_with_diag():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config(
        {
            "trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25},
            "testd": {"score_min": 0.0, "score_max": 0.25, "n_levels": 11},
        }
    )
    diagdict = {"testd": cfg.section("testd")}
    t_test_1 = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig), diag_configs=diagdict)
    t_test_1.advance()
    t_test_2 = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig), diag_configs=diagdict)
    t_test_2.advance()
    analyst = DiagnosticAnalyst("./diagDB.db")
    _ = analyst.get_diagnostic_data("testd")
    dstat = analyst.compute_weighted_stats("testd")
    assert dstat[0.0]["mean"] == 42.0
    _ = analyst.get_conditional_means("testd")
    analyst = None
    Path("./diagDB.db").unlink(missing_ok=True)


def test_branch_simple_model_traj():
    """Test branching a trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.0002, "targetscore": 0.45}})
    t_ancestor = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    t_ancestor.advance()
    assert t_ancestor.get_computed_steps_count() == 201
    t_branched = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig))
    t_branched = Trajectory.branch_from_trajectory(t_ancestor, t_branched, 0.1, 0.25)
    assert t_branched.get_computed_steps_count() == 0
    t_branched.advance()
    assert t_branched.get_computed_steps_count() == 150


def test_simple_model_traj_end_nstep():
    """Test trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"step_size": 0.001, "targetscore": 0.55}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(nstep_end=20)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    assert t_test.is_terminated() is True
    t_test.advance(t_end=0.03)
    assert isclose(t_test.score_max(), 0.3, abs_tol=1e-9)
    assert t_test.is_converged() is False


def test_store_and_restore_simple_traj():
    """Test store and restoring trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.05, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, cfg.load(TrajectoryConfig))
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkfile.unlink(missing_ok=True)


def test_store_and_restore_frozen_simple_traj():
    """Test store and restoring frozen trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.05, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.02)
    assert isclose(t_test.score_max(), 0.2, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, cfg.load(TrajectoryConfig), frozen=True)
    assert isclose(rst_test.score_max(), 0.2, abs_tol=1e-9)
    with pytest.raises(RuntimeError):
        rst_test.advance()
    with pytest.raises(RuntimeError):
        rst_test._one_step()
    chkfile.unlink(missing_ok=True)


def test_restart_simple_traj():
    """Test trajectory restart."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}})
    from_traj = Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig))
    from_traj.advance(t_end=0.01)
    rst_traj = Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig))
    rst_test = Trajectory.branch_from_trajectory(from_traj, rst_traj, 0.05, 0.25)
    assert rst_test.current_time() == 0.006


def test_access_data_simple_traj():
    """Test trajectory data access."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.01)
    assert t_test.get_length() == 11
    assert isclose(t_test.get_time_array()[-1], 0.01, abs_tol=1e-9)
    assert isclose(t_test.get_score_array()[-1], 0.1, abs_tol=1e-9)


def test_sparse_simple_traj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25, "sparse_freq": 5}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.012)
    assert isclose(t_test.score_max(), 0.12, abs_tol=1e-9)
    assert t_test.is_converged() is False
    assert isclose(t_test.get_last_state(), 0.01, abs_tol=1e-9)
    t_test.advance()
    assert t_test.is_converged() is True
    assert isclose(t_test.get_last_state(), 0.025, abs_tol=1e-9)


def test_sparse_simple_traj_access_states():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.0002, "targetscore": 0.25, "sparse_freq": 5}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance()
    assert len(t_test.get_state_list()) == 26


def test_store_and_restart_sparse_simple_traj():
    """Test a sparse trajectory with simple model."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.04, "step_size": 0.001, "targetscore": 0.25, "sparse_freq": 5}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance(t_end=0.013)
    assert isclose(t_test.score_max(), 0.13, abs_tol=1e-9)
    assert t_test.is_converged() is False
    chkfile = Path("./test.xml")
    t_test.store(chkfile)
    assert chkfile.exists() is True
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(chkfile, metadata, fmodel, cfg.load(TrajectoryConfig))
    rst_test.advance()
    assert rst_test.is_converged() is True
    chkfile.unlink()


def test_score_moving_average():
    """Test using a moving average on a score array."""
    fmodel = SimpleFModel
    cfg = Config({"trajectory": {"end_time": 0.9, "step_size": 0.0001, "targetscore": 0.95}})
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig))
    t_test.advance()
    score = t_test.get_score_array()
    avg_score = moving_avg(score, 10)
    assert isclose(avg_score[0], 0.0045, abs_tol=1e-9)


def test_sparse_dw_traj_with_restore():
    """Test restore a sparse trajectory with DW model."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 15.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10},
            "model": {"noise_amplitude": 0.8},
        }
    )
    model_params = cfg.section_dict("model")
    t_test = Trajectory(1, 1.0, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, deterministic=True)
    t_test.advance(t_end=4.07)
    chkfile = Path("./test.xml")
    t_test.store(chkfile, write_metadata_json=True)
    assert isclose(t_test.score_max(), 0.5384037112515893, abs_tol=1e-9)
    assert not t_test.is_converged()
    metadata = t_test.get_metadata()
    rst_test = Trajectory.restore_from_checkfile(
        chkfile, metadata, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, frozen=False
    )
    rst_test.advance()
    assert rst_test.score_max() > 0.95
    assert rst_test.is_converged()
    chkfile.unlink(missing_ok=True)
    Path("test.json").unlink(missing_ok=True)


def test_sparse_dw_traj_with_branching():
    """Test branching a sparse trajectory with simple model."""
    fmodel = DoubleWellModel
    cfg = Config(
        {
            "trajectory": {"end_time": 2.0, "step_size": 0.01, "targetscore": 0.95, "sparse_freq": 10},
            "model": {"noise_amplitude": 0.3},
        }
    )
    model_params = cfg.section_dict("model")
    t_test = [
        Trajectory(1, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, deterministic=True),
        Trajectory(2, 0.5, fmodel, cfg.load(TrajectoryConfig), model_params=model_params, deterministic=True),
    ]
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
