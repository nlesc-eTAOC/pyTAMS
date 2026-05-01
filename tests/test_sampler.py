"""Tests for the pytams.tams class."""

import logging
import shutil
from pathlib import Path
import pytest
import toml
from pytams.database import Database
from pytams.sampler import RareEventSampler
from pytams.utils import is_mac_os
from tests.dwmodel import DoubleWellModel
from tests.models import FailingFModel
from tests.models import SimpleFModel


def test_init_sampler():
    """Test sampler initialization."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "ams": {"ntrajectories": 500, "nsplititer": 200},
                "trajectory": {"end_time": 0.02, "step_size": 0.001}
            }, f
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    assert sampler.database.n_traj() == 500
    Path("input.toml").unlink(missing_ok=True)


def test_init_sampler_missing_req():
    """Test failed sampler initialization."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump({"sampler": {}, "tams": {"nsplititer": 200}, "trajectory": {"end_time": 0.02, "step_size": 0.001}}, f)
    with pytest.raises(ValueError):
        _ = RareEventSampler(fmodel_t=fmodel, a_args=[])
    Path("input.toml").unlink(missing_ok=True)


def test_init_sampler_no_input():
    """Test failed sampler initialization."""
    fmodel = SimpleFModel
    with pytest.raises(ValueError):
        _ = RareEventSampler(fmodel_t=fmodel, a_args=["-i", "dummy.toml"])


def test_simple_model_sampler():
    """Test sampler with simple model."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "WARNING"},
                "ams": {"ntrajectories": 100, "nsplititer": 200, "variant": "tams"},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba == 1.0
    Path("input.toml").unlink(missing_ok=True)


def test_simple_model_sampler_with_diags():
    """Test sampler with simple model and diags."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "WARNING", "diagnostics": ["testd"]},
                "ams": {"ntrajectories": 100, "nsplititer": 200, "variant": "tams"},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15},
                "testd": {"n_levels": 11},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba == 1.0
    Path("input.toml").unlink(missing_ok=True)
    Path("./diagDB.db").unlink(missing_ok=True)


def test_failing_model_sampler():
    """Test sampler with simple model."""
    fmodel = FailingFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams", "loglevel": "WARNING"},
                "ams": {"ntrajectories": 100, "nsplititer": 200},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 0.1, "step_size": 0.005, "targetscore": 0.75},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    Path("input.toml").unlink(missing_ok=True)
    with pytest.raises(RuntimeError):
        sampler.run()


def test_simple_model_init_ensemble_stage_tams(caplog: pytest.LogCaptureFixture):
    """Test sampler with tams and simple model."""
    caplog.set_level(logging.WARNING)
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "WARNING"},
                "ams": {"ntrajectories": 100, "nsplititer": 200, "init_ensemble_only": True},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 1.15},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    # Re-attach pytest handler for testing purposes
    logging.getLogger().addHandler(caplog.handler)
    sampler.run()
    assert "Stopping after the initial ensemble stage !" in caplog.text
    Path("input.toml").unlink(missing_ok=True)


def test_simple_model_init_ensemble_stage_and_continue_tams():
    """Test sampler with TAMS and simple model."""
    fmodel = SimpleFModel
    params_dict = {
        "sampler": {"strategy": "ams"},
        "runtime": {"loglevel": "INFO"},
        "ams": {"ntrajectories": 10, "nsplititer": 200, "init_ensemble_only": True},
        "runner": {"type": "asyncio"},
        "database": {"path": "simpleModelTest.tdb"},
        "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 1.15},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    del sampler
    tdb = Database.load(Path("simpleModelTest.tdb"))
    assert tdb.n_traj() == 10
    params_dict["ams"]["ntrajectories"] = 20
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    tdb.update_ntraj(20)
    assert tdb.n_traj() == 20

    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    del sampler
    tdb = Database.load(Path("simpleModelTest.tdb"))
    tdb.info()
    assert tdb.n_traj() == 20
    del tdb
    Path("input.toml").unlink(missing_ok=True)
    shutil.rmtree("simpleModelTest.tdb")


def test_simple_model_tams_with_db():
    """Test sampler with tams and simple model."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "WARNING"},
                "ams": {"ntrajectories": 100, "nsplititer": 200},
                "runner": {"type": "dask"},
                "database": {"path": "simpleModelTest.tdb"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15, "chkfile_dump_all": True},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    del sampler
    assert transition_proba == 1.0
    shutil.rmtree("simpleModelTest.tdb")
    Path("input.toml").unlink(missing_ok=True)


def test_simple_model_tams_with_db_access():
    """Test sampler with simple model and access to the database."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "WARNING"},
                "ams": {"ntrajectories": 100, "nsplititer": 200},
                "runner": {"type": "dask"},
                "database": {"path": "simpleModelTest.tdb"},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15, "chkfile_dump_all": True},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    tdb = sampler.database
    assert tdb.get_transition_probability() == 1
    del sampler
    del tdb
    shutil.rmtree("simpleModelTest.tdb")
    Path("input.toml").unlink(missing_ok=True)


def test_simple_model_mc_slurm_fail():
    """Test MonteCarlo with simple model with Slurm dask backend."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "montecarlo"},
                "runtime": {"loglevel": "DEBUG"},
                "montecarlo": {"ntrajectories": 100},
                "runner": {"type": "dask",
                           "dask": {"backend": "slurm", "slurm_config_file": "dummy.yaml"},
                           },
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    with pytest.raises(FileNotFoundError):
        sampler.run()
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_simple_model_twice_tams():
    """Test sampler with simple model."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "WARNING", "logfile": "test.log"},
                "ams": {"ntrajectories": 100, "nsplititer": 200},
                "runner": {"type": "asyncio"},
                "database": {"path": "simpleModelTest.tdb", "restart": True},
                "trajectory": {"end_time": 0.02, "step_size": 0.001, "targetscore": 0.15},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba == 1.0
    del sampler
    # Re-init TAMS and run to test competing database
    # on disk.
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba == 1.0
    del sampler
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
    """Test sampler with tams, simple model and stalled score function."""
    fmodel = SimpleFModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"loglevel": "ERROR"},
                "ams": {"ntrajectories": 100, "nsplititer": 200},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 1.0, "step_size": 0.01, "targetscore": 1.1},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    with pytest.raises(RuntimeError):
        sampler.run()


def test_sample_doublewell():
    """Test sampler with the doublewell model."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"walltime": 500.0},
                "ams": {"ntrajectories": 50, "nsplititer": 200},
                "runner": {"type": "dask"},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 6.0, "step_size": 0.01, "targetscore": 0.8},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba >= 0.2
    Path("input.toml").unlink(missing_ok=True)


def test_doublewell_save_tams():
    """Test sampler with TAMS on the doublewell model."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"walltime": 500.0},
                "ams": {"ntrajectories": 50, "nsplititer": 100},
                "runner": {"type": "dask"},
                "database": {"path": "dwTest.tdb"},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.3},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba >= 0.2
    del sampler
    Path("input.toml").unlink(missing_ok=True)
    shutil.rmtree("dwTest.tdb")


def test_doublewell_deterministic_tams():
    """Test TAMS with the doublewell model."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams", "deterministic": True},
                "runtime": {"walltime": 500.0},
                "ams": {"ntrajectories": 100, "nsplititer": 400},
                "runner": {"type": "asyncio"},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.8},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    if is_mac_os():
        assert transition_proba == 0.5416298076191378
    else:
        assert transition_proba == 0.5471008157769068
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.usefixtures("skip_on_windows")
def test_doublewell_deterministic_sampler_with_pltdiags(caplog: pytest.LogCaptureFixture):
    """Test sampler with tams on the doublewell model."""
    caplog.set_level(logging.WARNING)
    fmodel = DoubleWellModel
    Path("Score_k00001.png").touch()
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams", "deterministic": True},
                "runtime": {"walltime": 500.0, "plot_diagnostics": True},
                "ams": {
                    "ntrajectories": 5,
                    "nsplititer": 5,
                },
                "runner": {"type": "asyncio"},
                "model": {"noise_amplitude": 0.4},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.8},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    # Re-attach pytest handler for testing purposes
    logging.getLogger().addHandler(caplog.handler)
    _ = sampler.run()
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
                "sampler": {"strategy": "ams", "deterministic": True},
                "runtime": {"walltime": 500.0},
                "ams": {"ntrajectories": 50, "nsplititer": 400},
                "runner": {"type": "dask", "nworkers_init": 2, "nworkers_iter": 2},
                "model": {"noise_amplitude": 0.8},
                "database": {"path": "dwTest.tdb", "archive_discarded": True},
                "trajectory": {"end_time": 5.0, "step_size": 0.01, "targetscore": 0.5},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba == 0.6925339958244802
    del sampler
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
def test_doublewell_2_workers_restore_sampler():
    """Test TAMS with the doublewell model using two workers and restoring."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams", "deterministic": True},
                "runtime": {"walltime": 500.0},
                "ams": {"ntrajectories": 50, "nsplititer": 400},
                "runner": {"type": "dask", "nworkers_init": 2, "nworkers_iter": 2},
                "model": {"noise_amplitude": 0.8},
                "database": {"path": "dwTest.tdb", "archive_discarded": True},
                "trajectory": {"end_time": 5.0, "step_size": 0.01, "targetscore": 0.5},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba == 0.6925339958244802
    Path("input.toml").unlink(missing_ok=True)
    del sampler
    shutil.rmtree("dwTest.tdb")


def test_doublewell_very_slow_model():
    """Test sampler with tams run out of time with a slow doublewell."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"walltime": 3.0},
                "ams": {"ntrajectories": 10, "nsplititer": 400},
                "database": {"path": "vslowdwTest.tdb"},
                "runner": {"type": "dask"},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.7},
                "model": {"slow_factor": 0.01, "noise_amplitude": 0.1},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba <= 0.0
    Path("input.toml").unlink(missing_ok=True)
    del sampler
    shutil.rmtree("vslowdwTest.tdb")


@pytest.mark.dependency
def test_doublewell_slow_model_stop():
    """Test sampler run out of time with a slow doublewell."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"walltime": 3.0},
                "ams": {"ntrajectories": 10, "nsplititer": 400},
                "database": {"path": "slowdwTest.tdb"},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 8.0, "step_size": 0.01, "targetscore": 0.9},
                "model": {"slow_factor": 0.0005, "noise_amplitude": 0.1},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba <= 0.0
    del sampler
    Path("input.toml").unlink(missing_ok=True)


@pytest.mark.dependency(depends=["test_doublewell_slow_model_stop"])
def test_doublewell_slow_tams_restore_during_initial_ensemble():
    """Test TAMS restarting a slow doublewell."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "sampler": {"strategy": "ams"},
                "runtime": {"walltime": 8.0, "loglevel": "INFO"},
                "ams": {"ntrajectories": 10, "nsplititer": 400},
                "database": {"path": "slowdwTest.tdb"},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 8.0, "step_size": 0.01, "targetscore": 0.9},
                "model": {"slow_factor": 0.0005, "noise_amplitude": 0.1},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    assert transition_proba <= 0.0
    del sampler
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
                "sampler": {"strategy": "ams"},
                "runtime": {"walltime": 2.0, "loglevel": "INFO"},
                "ams": {"ntrajectories": 10, "nsplititer": 400},
                "database": {"path": "slowdwTest.tdb"},
                "runner": {"type": "asyncio"},
                "trajectory": {"end_time": 8.0, "step_size": 0.01, "targetscore": 0.9},
                "model": {"slow_factor": 0.0005, "noise_amplitude": 0.1},
            },
            f,
        )
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    # Re-attach pytest handler for testing purposes
    logging.getLogger().addHandler(caplog.handler)
    _ = sampler.run()
    assert "Unfinished splitting iteration detected" in caplog.text
    Path("input.toml").unlink(missing_ok=True)
    del sampler
    shutil.rmtree("slowdwTest.tdb")


@pytest.mark.usefixtures("skip_on_windows")
def test_doublewell_slow_tams_restore_more_split():
    """Test restart TAMS more splitting iterations."""
    fmodel = DoubleWellModel
    params_dict = {
        "sampler": {"strategy": "ams", "deterministic": True},
        "runtime": {"walltime": 20.0},
        "ams": {"ntrajectories": 20, "nsplititer": 20},
        "database": {"path": "dwTest.tdb"},
        "runner": {"type": "asyncio", "nworkers_init": 2, "nworkers_iter": 1},
        "trajectory": {"end_time": 6.0, "step_size": 0.01, "targetscore": 0.6},
        "model": {"slow_factor": 0.00000001, "noise_amplitude": 0.6},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    del sampler
    assert transition_proba == 0.1251225103143388
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.update_nsplititer(30)
    params_dict["ams"]["nsplititer"] = 30
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    del tdb

    sampler = RareEventSampler(fmodel_t=fmodel, a_args=[])
    sampler.run()
    transition_proba = sampler.database.get_transition_probability()
    # Not sure why this particular test is platform dependent
    if is_mac_os():
        assert transition_proba == 0.1391287278743694
    else:
        assert transition_proba == 0.14983093771085937
    Path("input.toml").unlink(missing_ok=True)
    del sampler
    shutil.rmtree("dwTest.tdb")
