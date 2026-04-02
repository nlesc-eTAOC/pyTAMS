"""Tests for the pytams.diagnostics class."""

import pickle
from pathlib import Path
import pytest
from pytams.diagdb import DiagDB
from pytams.diagnostic import diagnosticfactory
from pytams.snapshot import Snapshot
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
    params_dict = {"tams": {"diagnostics": ["testd"]}, "testd": {"type": "FirstCrossing"}}
    dplugin = diagnosticfactory(params_dict, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    assert dplugin[0]._label == "testd"
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_failed_init_diagnostic():
    """Test initialize a diagnostic."""
    ddb = DiagDB("./diagDBtest.db")
    params_dict = {"tams": {"diagnostics": ["testd"]}}
    with pytest.raises(RuntimeError):
        _ = diagnosticfactory(params_dict, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_init_diagnostic_with_levels():
    """Test initialize a diagnostic."""
    ddb = DiagDB("./diagDBtest.db")
    params_dict = {"tams": {"diagnostics": ["testd"]}, "testd": {"type": "FirstCrossing", "n_levels": 3}}
    dplugin = diagnosticfactory(params_dict, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    assert dplugin[0]._levels[1] == 0.5
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_diagnostic_crossed():
    """Test initialize a diagnostic and crossed."""
    ddb = DiagDB("./diagDBtest.db")
    params_dict = {"tams": {"diagnostics": ["testd"]}, "testd": {"type": "FirstCrossing", "n_levels": 3}}
    dplugin = diagnosticfactory(params_dict, 1, 0.1, "./", SimpleFModel.diagnostic_hook, ddb)
    s_new = Snapshot(time=1.0, score=0.6, noise=0.0)
    levels = dplugin[0].get_crossed_levels(s_new)
    assert len(levels) == 2
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)


def test_diagnostic_update():
    """Test initialize a diagnostic and update."""
    fmodel = SimpleFModel(1, {})
    ddb = DiagDB("./diagDBtest.db")
    params_dict = {"tams": {"diagnostics": ["testd"]}, "testd": {"type": "FirstCrossing", "n_levels": 3}}
    dplugin = diagnosticfactory(params_dict, 1, 0.1, "./", fmodel.diagnostic_hook, ddb)
    s_old = Snapshot(time=0.0, score=0.0, noise=0.0)
    s_new = Snapshot(time=1.0, score=0.6, noise=0.0)
    dplugin[0].update(s_old, s_new)
    data = ddb.get_diagnostic_data("testd")
    assert data[0.0] == [(42.0, 0.1, 1.0, 1)]
    dplugin = []
    del ddb
    Path("./diagDBtest.db").unlink(missing_ok=True)
