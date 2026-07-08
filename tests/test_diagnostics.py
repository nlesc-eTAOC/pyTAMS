"""Tests for the pyrevs.diagnostics class."""

import pickle
from pathlib import Path
import pytest
from pyrevs.core import Config
from pyrevs.core import Snapshot
from pyrevs.diagnostics import DiagDB
from pyrevs.diagnostics import diagnosticfactory
from tests.models import SimpleFModel


def test_init_diagdb():
    """Test initialize a diagnostic database."""
    ddb = DiagDB("diagDBtest.db")
    assert ddb.name() == "diagDBtest.db"
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_add_to_diagdb():
    """Test initialize a diagnostic database and add."""
    ddb = DiagDB("diagDBtest.db")
    ddb.add_diagnostic_entry(
        diaglabel="testd",
        traj_id=1,
        level=0.1,
        time=1.0,
        weight=0.1,
        ldata=pickle.dumps([0.0, 1.0]),
    )
    nb_update = ddb.update_all_active_weights(0.01)
    assert nb_update == 1
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_init_diagnostic():
    """Test initialize a diagnostic."""
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "FirstCrossing"})}
    dplugin = diagnosticfactory(dconfig, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    assert dplugin[0]._label == "testd"
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_init_unknown_diagnostic():
    """Test initialize a diagnostic."""
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "Unknown"})}
    with pytest.raises(ValueError):
        _ = diagnosticfactory(dconfig, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_init_diagnostic_with_levels():
    """Test initialize a diagnostic."""
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "FirstCrossing", "n_levels": 3})}
    dplugin = diagnosticfactory(dconfig, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    assert dplugin[0]._levels[1] == 0.5
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_diagnostic_crossed():
    """Test initialize a diagnostic and crossed."""
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "FirstCrossing", "n_levels": 3})}
    dplugin = diagnosticfactory(dconfig, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    s_new = Snapshot(time=1.0, score=0.6, noise=0.0)
    levels = dplugin[0].get_crossed_levels(s_new)
    assert len(levels) == 2
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_multiple_diagnostic_crossed():
    """Test initialize two diagnostic and crossed."""
    fmodel = SimpleFModel(1, {})
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "FirstCrossing", "n_levels": 3}),
               "testd2": Config({"type": "FirstCrossing", "n_levels": 3})}
    dplugin = diagnosticfactory(dconfig, 1, 0.1, "./", fmodel.diagnostic_hook, ddb)
    s_old = Snapshot(time=0.0, score=0.0, noise=0.0)
    s_new = Snapshot(time=1.0, score=0.6, noise=0.0)
    dplugin[0].update(s_old, s_new)
    dplugin[1].update(s_old, s_new)
    assert ddb.count_entries() == 4
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)

def test_multiple_traj_diagnostic_crossed():
    """Test initialize a diagnostic and crossed by two trajectories."""
    fmodel = SimpleFModel(1, {})
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "FirstCrossing", "n_levels": 3})}
    dplugin1 = diagnosticfactory(dconfig, 1, 0.1, "./", fmodel.diagnostic_hook, ddb)
    dplugin2 = diagnosticfactory(dconfig, 2, 0.1, "./", fmodel.diagnostic_hook, ddb)
    s_old = Snapshot(time=0.0, score=0.0, noise=0.0)
    s_new = Snapshot(time=1.0, score=0.6, noise=0.0)
    dplugin1[0].update(s_old, s_new)
    dplugin2[0].update(s_old, s_new)
    assert ddb.get_unique_traj_ids() == [1, 2]
    dplugin1 = []
    dplugin2 = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_diagnostic_update():
    """Test initialize a diagnostic and update."""
    fmodel = SimpleFModel(1, {})
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "FirstCrossing", "n_levels": 3})}
    dplugin = diagnosticfactory(dconfig, 1, 0.1, "./", fmodel.diagnostic_hook, ddb)
    s_old = Snapshot(time=0.0, score=0.0, noise=0.0)
    s_new = Snapshot(time=1.0, score=0.6, noise=0.0)
    dplugin[0].update(s_old, s_new)
    data = ddb.get_diagnostic_data("testd")
    assert data[0.0] == [(42.0, 0.1, 0.0, 1)]
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)

def test_diagnostic_duplicate_history():
    """Test initialize a diagnostic, update and duplicate."""
    fmodel = SimpleFModel(1, {})
    ddb = DiagDB("./diagDBtest.db")
    dconfig = {"testd": Config({"type": "FirstCrossing", "n_levels": 10})}
    dplugin_disc = diagnosticfactory(dconfig, 1, 0.1, "./", fmodel.diagnostic_hook, ddb)
    dplugin_anc = diagnosticfactory(dconfig, 2, 0.1, "./", fmodel.diagnostic_hook, ddb)
    s_old = Snapshot(time=0.0, score=0.0, noise=0.0)
    s_new = Snapshot(time=1.0, score=0.6, noise=0.0)
    dplugin_disc[0].update(s_old, s_new)
    dplugin_anc[0].update(s_old, s_new)
    ddb.duplicate_diagnostic_history_from_time(1, 2, 3, 0.05, 0.2)
    data = ddb.get_diagnostic_data("testd")
    assert data[0.0] == [(42.0, 0.1, 0.0, 1), (42.0, 0.1, 0.0, 2), (42.0, 0.05, 0.0, 3)]
    dplugin_disc = []
    dplugin_anc = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)
