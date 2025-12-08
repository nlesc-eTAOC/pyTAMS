"""Tests for the pytams.tams class."""

import logging
import shutil
from pathlib import Path
import pytest
import toml
from pytams.database import Database
from pytams.tams import TAMS
from pytams.utils import is_mac_os
from tests.models import DoubleWellModel
from tests.models import SimpleFModel


def test_init_tams():
    """Test TAMS initialization."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {"tams": {"ntrajectories": 500, "nsplititer": 200}, "trajectory": {"end_time": 0.02, "step_size": 0.001}}, f
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    assert tams.n_traj() == 500
    Path("input.toml").unlink(missing_ok=True)


def test_init_tams_missing_req():
    """Test failed TAMS initialization."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump({"tams": {"nsplititer": 200}, "trajectory": {"end_time": 0.02, "step_size": 0.001}}, f)
    with pytest.raises(ValueError):
        _ = TAMS(fmodel_t=fmodel, a_args=[])
    Path("input.toml").unlink(missing_ok=True)


def test_init_tams_no_input():
    """Test failed TAMS initialization."""
    fmodel = SimpleFModel
    with pytest.raises(ValueError):
        _ = TAMS(fmodel_t=fmodel, a_args=["-i", "dummy.toml"])


def test_simple_model_tams():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING"},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0
    Path("input.toml").unlink(missing_ok=True)


def test_simple_model_init_ensemble_stage_tams(caplog: pytest.LogCaptureFixture):
    """Test TAMS with simple model."""
    caplog.set_level(logging.WARNING)
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING", "init_ensemble_only": True},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 1.15},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    _ = tams.compute_probability()
    assert "Stopping after the initial ensemble stage !" in caplog.text
    Path("input.toml").unlink(missing_ok=True)


def test_simple_model_init_ensemble_stage_and_continue_tams():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    params_dict = {
        "tams": {"ntrajectories": 10, "nsplititer": 200, "loglevel": "WARNING", "init_ensemble_only": True},
        "runner": {"type": "asyncio"},
        "database": {"path": "simpleModelTest.tdb"},
        "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 1.15},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    _ = tams.compute_probability()
    del tams
    tdb = Database.load(Path("simpleModelTest.tdb"))
    assert tdb.n_traj() == 10
    del tdb
    params_dict["tams"]["ntrajectories"] = 20
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    _ = tams.compute_probability()
    del tams
    tdb = Database.load(Path("simpleModelTest.tdb"))
    assert tdb.n_traj() == 20
    del tdb
    Path("input.toml").unlink(missing_ok=True)
    shutil.rmtree("simpleModelTest.tdb")


def test_simple_model_tams_with_db():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING"},
                "runner": {"type": "dask"},
                "database": {"path": "simpleModelTest.tdb"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15, "chkfile_dump_all": True},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    del tams
    assert transition_proba == 1.0
    shutil.rmtree("simpleModelTest.tdb")
    Path("input.toml").unlink(missing_ok=True)

def test_simple_model_tams_with_db_access():
    """Test TAMS with simple model and access to the database."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING"},
                "runner": {"type": "dask"},
                "database": {"path": "simpleModelTest.tdb"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15, "chkfile_dump_all": True},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    _ = tams.compute_probability()
    tdb = tams.get_database()
    assert tdb.get_transition_probability() == 1
    del tams
    del tdb
    shutil.rmtree("simpleModelTest.tdb")
    Path("input.toml").unlink(missing_ok=True)

def test_simple_model_tams_slurm_fail():
    """Test TAMS with simple model with Slurm dask backend."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "DEBUG"},
                "runner": {"type": "dask"},
                "dask": {"backend": "slurm", "slurm_config_file": "dummy.yaml"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    with pytest.raises(FileNotFoundError):
        tams.compute_probability()
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_simple_model_twice_tams():
    """Test TAMS with simple model."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "WARNING", "logfile": "test.log"},
                "runner": {"type": "asyncio"},
                "database": {"path": "simpleModelTest.tdb", "restart": True},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 1.0
    del tams
    # Re-init TAMS and run to test competing database
    # on disk.
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    del tams
    ndb = 0
    for folder in Path("./").iterdir():
        if "simpleModelTest" in str(folder):
            shutil.rmtree(folder)
            ndb += 1
    assert ndb == 2
    assert Path("test.log").exists()
    Path("test.log").unlink(missing_ok=True)
    Path("input.toml").unlink(missing_ok=True)


def test_stalling_simplemodel_tams():
    """Test TAMS with simple model and stalled score function."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 200, "loglevel": "ERROR"},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 1.0, "step_size": 0.01, "targetscore": 1.1},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    with pytest.raises(RuntimeError):
        tams.compute_probability()


def test_doublewell_tams():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 50, "nsplititer": 200, "walltime": 500.0},
                "runner": {"type": "dask"},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 6.0, "step_size": 0.01, "targetscore": 0.8},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2
    Path("input.toml").unlink(missing_ok=True)


def test_doublewell_save_tams():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 50, "nsplititer": 100, "walltime": 500.0},
                "runner": {"type": "dask"},
                "database": {"path": "dwTest.tdb"},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.3},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2
    del tams
    Path("input.toml").unlink(missing_ok=True)
    shutil.rmtree("dwTest.tdb")


def test_doublewell_deterministic_tams():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 400, "walltime": 500.0, "deterministic": True},
                "runner": {"type": "asyncio"},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.8},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 0.5471567691116268
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_doublewell_deterministic_tams_with_diags(caplog: pytest.LogCaptureFixture):
    """Test TAMS with the doublewell model."""
    caplog.set_level(logging.WARNING)
    fmodel = DoubleWellModel
    Path("Score_k00001.png").touch()
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {
                    "ntrajectories": 5,
                    "nsplititer": 5,
                    "walltime": 500.0,
                    "deterministic": True,
                    "diagnostics": True,
                },
                "runner": {"type": "asyncio"},
                "model": {"noise_amplitude": 0.4},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.8},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    _ = tams.compute_probability()
    assert "Attempting to overwrite the plot file" in caplog.text
    Path("input.toml").unlink(missing_ok=True)
    for p in Path().glob("Score*.png"):
        p.unlink()


@pytest.mark.dependency
def test_doublewell_2_workers_tams():
    """Test TAMS with the doublewell model using two workers."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 50, "nsplititer": 400, "walltime": 500.0, "deterministic": True},
                "runner": {"type": "dask", "nworker_init": 2, "nworker_iter": 2},
                "model": {"noise_amplitude": 0.8},
                "database": {"path": "dwTest.tdb", "archive_discarded": True},
                "trajectory": {"end_time": 5.0, "step_size": 0.01, "targetscore": 0.5},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba == 0.692533980184018
    del tams
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.dependency(depends=["test_doublewell_2_workers_tams"])
def test_doublewell_2_workers_load_db():
    """Load the database from previous test."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.load_data(True)
    assert tdb.traj_list_len() == 50
    assert tdb.archived_traj_list_len() == 16
    del tdb


@pytest.mark.dependency(depends=["test_doublewell_2_workers_tams"])
def test_doublewell_2_workers_restore_tams():
    """Test TAMS with the doublewell model using two workers and restoring."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 100, "nsplititer": 400, "walltime": 500.0},
                "database": {"path": "dwTest.tdb"},
                "runner": {"type": "asyncio", "nworker_init": 2, "nworker_iter": 2},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.6},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba >= 0.2
    Path("input.toml").unlink(missing_ok=True)
    del tams
    shutil.rmtree("dwTest.tdb")


def test_doublewell_very_slow_tams():
    """Test TAMS run out of time with a slow doublewell."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 10, "nsplititer": 400, "walltime": 3.0},
                "database": {"path": "vslowdwTest.tdb"},
                "runner": {"type": "dask", "nworker_init": 1, "nworker_iter": 1},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.7},
                "model": {"slow_factor": 0.01, "noise_amplitude": 0.1},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba <= 0.0
    Path("input.toml").unlink(missing_ok=True)
    del tams
    shutil.rmtree("vslowdwTest.tdb")


@pytest.mark.dependency
def test_doublewell_slow_tams_stop():
    """Test TAMS run out of time with a slow doublewell."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 8, "nsplititer": 400, "walltime": 3.0},
                "database": {"path": "slowdwTest.tdb"},
                "runner": {"type": "asyncio", "nworker_init": 1, "nworker_iter": 1},
                "trajectory": {"end_time": 8.0, "step_size": 0.01, "targetscore": 0.9},
                "model": {"slow_factor": 0.0002, "noise_amplitude": 0.1},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba <= 0.0
    del tams
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.dependency(depends=["test_doublewell_slow_tams_stop"])
def test_doublewell_slow_tams_restore_during_initial_ensemble():
    """Test TAMS restarting a slow doublewell."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 8, "nsplititer": 400, "walltime": 8.0},
                "database": {"path": "slowdwTest.tdb"},
                "runner": {"type": "asyncio", "nworker_init": 1, "nworker_iter": 1},
                "trajectory": {"end_time": 8.0, "step_size": 0.01, "targetscore": 0.9},
                "model": {"slow_factor": 0.0002, "noise_amplitude": 0.1},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    assert transition_proba <= 0.0
    del tams
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
@pytest.mark.dependency(depends=["test_doublewell_slow_tams_restore_during_initial_ensemble"])
def test_doublewell_slow_tams_restore_during_splitting(caplog: pytest.LogCaptureFixture):
    """Test TAMS restarting a slow doublewell."""
    caplog.set_level(logging.INFO)
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 8, "nsplititer": 400, "walltime": 2.0},
                "database": {"path": "slowdwTest.tdb"},
                "runner": {"type": "asyncio", "nworker_init": 1, "nworker_iter": 1},
                "trajectory": {"end_time": 8.0, "step_size": 0.01, "targetscore": 0.9},
                "model": {"slow_factor": 0.0002, "noise_amplitude": 0.1},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    _ = tams.compute_probability()
    assert "Unfinished splitting iteration detected" in caplog.text
    Path("input.toml").unlink(missing_ok=True)
    del tams
    shutil.rmtree("slowdwTest.tdb")


@pytest.mark.usefixtures("skip_on_windows")
def test_doublewell_slow_tams_restore_more_split():
    """Test restart TAMS more splitting iterations."""
    fmodel = DoubleWellModel
    params_dict = {
        "tams": {"ntrajectories": 20, "nsplititer": 20, "walltime": 20.0, "deterministic": True},
        "database": {"path": "dwTest.tdb"},
        "runner": {"type": "asyncio", "nworker_init": 2, "nworker_iter": 1},
        "trajectory": {"end_time": 6.0, "step_size": 0.01, "targetscore": 0.6},
        "model": {"slow_factor": 0.00000001, "noise_amplitude": 0.6},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams.compute_probability()
    del tams
    assert transition_proba == 0.08937321391676148
    params_dict["tams"]["nsplititer"] = 30
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    tams_load = TAMS(fmodel_t=fmodel, a_args=[])
    transition_proba = tams_load.compute_probability()
    # Not sure why this particular test is platform dependent
    if is_mac_os():
        assert transition_proba == 0.0853804929982172
    else:
        assert transition_proba == 0.07470793360861466
    Path("input.toml").unlink(missing_ok=True)
    del tams_load
    shutil.rmtree("dwTest.tdb")
