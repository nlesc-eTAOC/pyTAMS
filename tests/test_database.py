"""Tests for the pytams.database class."""

import shutil
from pathlib import Path
import pytest
import toml
from pytams.database import Database
from pytams.tams import TAMS
from tests.models import DoubleWellModel


def test_failed_db_init_no_ntraj():
    """Test init of TDB from scratch missing argument."""
    fmodel = DoubleWellModel
    params_load_db = {}
    with pytest.raises(ValueError):
        Database(fmodel, params_load_db)

def test_failed_db_init_no_nsplit():
    """Test init of TDB failing missing nsplit."""
    fmodel = DoubleWellModel
    params_load_db = {}
    with pytest.raises(ValueError):
        Database(fmodel, params_load_db, ntraj=10)

def test_wrong_format():
    """Test init of TDB with unsupported format."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb", "format": "WRONG"}}
    with pytest.raises(ValueError):
        _ = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)

def test_load_missing_tdb():
    """Test failed load database."""
    with pytest.raises(FileNotFoundError):
        _ = Database.load(Path("dwTestNonExistent.tdb"))

def test_init_empty_tdb_inmemory():
    """Test init database."""
    fmodel = DoubleWellModel
    params_load_db = {}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    assert tdb.name() == "TAMS_DoubleWellModel"

def test_init_empty_tdb():
    """Test init database on disk."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    assert tdb.name() == "dwTest.tdb"
    # Necessary on Windows
    del tdb
    shutil.rmtree("dwTest.tdb")

def test_reinit_empty_tdb():
    """Test init database on disk."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTestDouble.tdb"}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    # Necessary on Windows
    del tdb
    params_load_db = {"database": {"path": "dwTestDouble.tdb", "restart": True}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    del tdb
    ndb = 0
    for folder in Path("./").iterdir():
        if "dwTestDouble" in str(folder):
            shutil.rmtree(folder)
            ndb += 1
    assert ndb == 2

def test_init_and_load_empty_tdb():
    """Test init database on disk."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db, ntraj=10, nsplititer=100)
    tdb_path = Path(tdb.path())
    assert tdb.name() == "dwTest.tdb"
    del tdb
    tdb = Database.load(tdb_path)
    del tdb
    shutil.rmtree("dwTest.tdb")

@pytest.mark.dependency(name="genDB")
def test_generate_and_load_tdb():
    """Test generation of TDB and loading the TDB."""
    fmodel = DoubleWellModel
    with Path("input.toml").open("w") as f:
        toml.dump(
            {
                "tams": {"ntrajectories": 50, "nsplititer": 200, "walltime": 500.0, "loglevel": "DEBUG"},
                "database": {"path": "dwTest.tdb"},
                "runner": {"type": "asyncio", "nworker_init": 2, "nworker_iter": 1},
                "model": {"noise_amplitude": 0.8},
                "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.51},
            },
            f,
        )
    tams = TAMS(fmodel_t=fmodel, a_args=[])
    tams.compute_probability()

    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.info()
    assert tdb
    Path("input.toml").unlink(missing_ok=True)

@pytest.mark.dependency(depends=["genDB"])
def test_access_ensemble_length():
    """Test accessing database trajectory ensemble size."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb.is_empty() is False

@pytest.mark.dependency(depends=["genDB"])
def test_access_ended_count():
    """Test accessing database trajectory metadata."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb.count_ended_traj() == 50

@pytest.mark.dependency(depends=["genDB"])
def test_access_converged_count():
    """Test accessing database trajectory metadata."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb.count_converged_traj() == 50

@pytest.mark.dependency(depends=["genDB"])
def test_access_pool_content():
    """Test accessing database SQL pool content."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.get_pool_db().dump_file_json("test.json")
    assert Path("./test.json").exists()
    Path("./test.json").unlink()

@pytest.mark.dependency(depends=["genDB"])
def test_copy_and_access():
    """Test copying the database and accessing it."""
    shutil.copytree("dwTest.tdb", "dwTestCopy.tdb")
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTestCopy.tdb"}}
    tdb = Database(fmodel, params_load_db)
    assert tdb.count_converged_traj() == 50
    del tdb
    shutil.rmtree("dwTestCopy.tdb")

@pytest.mark.dependency(depends=["genDB"])
def test_replace_traj_in_tdb():
    """Test replacing a trajectory in the database."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()

    traj_zero = tdb.get_traj(0)
    tdb.overwrite_traj(1, traj_zero)
    assert tdb.get_traj(1).idstr()[:10] == "traj000000"

@pytest.mark.dependency(depends=["genDB"])
def test_unknown_traj_access_in_tdb():
    """Test accessing a trajectory out-of-range."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()

    with pytest.raises(ValueError):
        _ = tdb.get_traj(10000)

@pytest.mark.dependency(depends=["genDB"])
def test_unknown_traj_overwrite_in_tdb():
    """Test overwriting a trajectory out-of-range."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()

    traj_zero = tdb.get_traj(0)
    with pytest.raises(ValueError):
        _ = tdb.overwrite_traj(10000, traj_zero)

@pytest.mark.dependency(depends=["genDB"])
def test_access_trajdata_in_tdb():
    """Test accessing a trajectory in the database."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()

    traj = tdb.get_traj(0)
    times = traj.get_time_array()
    scores = traj.get_score_array()
    noises = traj.get_noise_array()
    assert times.size > 0
    assert scores.size > 0
    assert noises.size > 0

@pytest.mark.dependency(depends=["genDB"])
def test_explore_tdb():
    """Test loading the TDB."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()
    tdb.plot_score_functions("test.png")
    Path("./test.png").unlink(missing_ok=False)

@pytest.mark.dependency(depends=["genDB"])
def test_explore_minmax_tdb():
    """Test loading the TDB."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db)
    tdb.load_data()
    tdb.plot_min_max_span(fname="test_minmax.png")
    Path("./test_minmax.png").unlink(missing_ok=False)

@pytest.mark.dependency(depends=["genDB"])
def test_explore_active_at_k():
    """Test getting the initial active set."""
    tdb = Database.load(Path("dwTest.tdb"))
    tdb.load_data(load_archived_trajectories=True)
    act_trajs = tdb.get_trajectory_active_at_k(0)
    assert len(act_trajs) == 50
    assert act_trajs[42].idstr() == "traj000042_0000"

@pytest.mark.dependency(depends=["genDB"])
def test_restore_tdb():
    """Test loading and restoring the TDB."""
    fmodel = DoubleWellModel
    params_load_db = {"database": {"path": "dwTest.tdb"}}
    tdb = Database(fmodel, params_load_db, read_only=False)
    tdb.load_data()
    tdb.reset_initial_ensemble_stage()
    assert tdb.k_split() == 0
    del tdb
    shutil.rmtree("dwTest.tdb")
