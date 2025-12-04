"""Tests for the pytams.worker functions."""

import datetime
from math import isclose
from pathlib import Path
import pytest
from pytams.sqldb import SQLFile
from pytams.trajectory import Trajectory
from pytams.utils import setup_logger
from pytams.worker import ms_worker
from pytams.worker import pool_worker
from tests.models import DoubleWellModel
from tests.models import FailingFModel
from tests.models import SimpleFModel


def test_run_pool_worker():
    """Advance trajectory through pool_worker."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}}
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    t_test = pool_worker(t_test, enddate)
    assert isclose(t_test.score_max(), 0.1, abs_tol=1e-9)
    assert t_test.is_converged() is False


def test_run_pool_worker_with_sql():
    """Advance trajectory through pool_worker with SQL."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}}
    poolfile = SQLFile("./test.db")
    t_test = Trajectory(0, 1.0, fmodel, parameters)
    poolfile.add_trajectory("dummy.xml", t_test.serialize_metadata_json())
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    _ = pool_worker(t_test, enddate, "./test.db")
    _, metadata_str = poolfile.fetch_trajectory(0)
    assert Trajectory.deserialize_metadata(metadata_str)["ended"]
    del poolfile
    Path("./test.db").unlink()


def test_run_pool_worker_outoftime(caplog: pytest.LogCaptureFixture):
    """Advance trajectory through pool_worker running out of time."""
    fmodel = DoubleWellModel
    parameters = {
        "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.75},
        "tams": {"loglevel": "DEBUG"},
        "model": {"slow_factor": 0.03},
    }
    setup_logger(parameters)
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=0.1)
    _ = pool_worker(t_test, enddate)
    assert "advance ran out of time" in caplog.text


def test_run_pool_worker_advanceerror():
    """Advance trajectory through pool_worker running into error."""
    fmodel = FailingFModel
    parameters = {
        "trajectory": {"end_time": 1.0, "step_size": 0.01, "targetscore": 0.75},
        "tams": {"loglevel": "DEBUG"},
    }
    setup_logger(parameters)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=1.0)
    t_test = Trajectory(1, 1.0, fmodel, parameters)
    with pytest.raises(RuntimeError):
        _ = pool_worker(t_test, enddate)


def test_run_ms_worker():
    """Branch and advance trajectory through ms_worker."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}}
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    t_test = Trajectory(1, 0.5, fmodel, parameters)
    t_test.advance()
    rst_test = Trajectory(2, 0.5, fmodel, parameters)
    b_test = ms_worker(t_test, rst_test, 0.049, 1.0, enddate)
    assert b_test.id() == 2
    assert isclose(b_test.score_max(), 0.1, abs_tol=1e-9)
    assert b_test.is_converged() is False


def test_run_ms_worker_with_sql():
    """Branch and advance trajectory through ms_worker."""
    fmodel = SimpleFModel
    parameters = {"trajectory": {"end_time": 0.01, "step_size": 0.001, "targetscore": 0.25}}
    poolfile = SQLFile("./test.db")
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=10.0)
    t_test = Trajectory(0, 0.5, fmodel, parameters)
    t_test.advance()
    poolfile.add_trajectory("dummy.xml", t_test.serialize_metadata_json())
    rst_test = Trajectory(1, 0.5, fmodel, parameters)
    poolfile.add_trajectory("dummy.xml", rst_test.serialize_metadata_json())
    _ = ms_worker(t_test, rst_test, 0.049, 1.0, enddate, "./test.db")
    _, metadata_str = poolfile.fetch_trajectory(1)
    assert Trajectory.deserialize_metadata(metadata_str)["ended"]
    del poolfile
    Path("./test.db").unlink()


def test_run_ms_worker_model_outoftime(caplog: pytest.LogCaptureFixture):
    """Advance trajectory through pool_worker running out of time."""
    fmodel = DoubleWellModel
    parameters = {
        "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.75},
        "tams": {"loglevel": "DEBUG"},
        "model": {"slow_factor": 0.003},
    }
    setup_logger(parameters)
    t_test = Trajectory(1, 0.5, fmodel, parameters)
    t_test.advance()
    rst_test = Trajectory(2, 0.5, fmodel, parameters)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=0.1)
    _ = ms_worker(t_test, rst_test, 0.1, 1.0, enddate)
    assert "advance ran out of time" in caplog.text


def test_run_ms_worker_outoftime(caplog: pytest.LogCaptureFixture):
    """Advance trajectory through pool_worker running out of time."""
    fmodel = DoubleWellModel
    parameters = {
        "trajectory": {"end_time": 10.0, "step_size": 0.01, "targetscore": 0.75},
        "tams": {"loglevel": "DEBUG"},
        "model": {"slow_factor": 0.003},
    }
    setup_logger(parameters)
    t_test = Trajectory(1, 0.5, fmodel, parameters)
    t_test.advance()
    rst_test = Trajectory(2, 0.5, fmodel, parameters)
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=0.1)
    _ = ms_worker(t_test, rst_test, 0.1, 1.0, enddate)
    assert "MS worker ran out of time" in caplog.text


def test_run_ms_worker_advanceerror():
    """Advance trajectory through pool_worker running into error."""
    fmodel = FailingFModel
    parameters = {
        "trajectory": {"end_time": 1.0, "step_size": 0.001, "targetscore": 0.75},
        "tams": {"loglevel": "DEBUG"},
    }
    enddate = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=1.0)
    setup_logger(parameters)
    t_test = Trajectory(1, 0.5, fmodel, parameters)
    t_test.advance(0.01)
    rst_test = Trajectory(5, 0.5, fmodel, parameters)
    with pytest.raises(RuntimeError):
        _ = ms_worker(t_test, rst_test, 0.04, 1.0, enddate)
