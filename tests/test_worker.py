"""Tests for the pytams.worker functions."""

import time
from math import isclose
import pytest
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
    parameters = {"trajectory" : {"end_time": 0.01,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, 1)
    t_test = pool_worker(t_test, time.monotonic() + 10.0, False, "testDB")
    assert isclose(t_test.scoreMax(), 0.1, abs_tol=1e-9)
    assert t_test.isConverged() is False

def test_run_pool_worker_outoftime(caplog : pytest.LogCaptureFixture):
    """Advance trajectory through pool_worker running out of time."""
    fmodel = DoubleWellModel
    parameters = {"trajectory" : {"end_time": 10.0,
                                  "step_size": 0.01,
                                  "targetscore": 0.75},
                  "tams": {"loglevel": "DEBUG"},
                  "model": {"slow_factor": 0.003}}
    setup_logger(parameters)
    t_test = Trajectory(fmodel, parameters, 1)
    _ = pool_worker(t_test, time.monotonic() + 0.1, False, "testDB")
    assert "advance ran out of time" in caplog.text

def test_run_pool_worker_advanceerror():
    """Advance trajectory through pool_worker running into error."""
    fmodel = FailingFModel
    parameters = {"trajectory" : {"end_time": 1.0,
                                  "step_size": 0.01,
                                  "targetscore": 0.75},
                  "tams": {"loglevel": "DEBUG"}}
    setup_logger(parameters)
    t_test = Trajectory(fmodel, parameters, 1)
    with pytest.raises(RuntimeError):
        _ = pool_worker(t_test, time.monotonic() + 1.0, False, "testDB")

def test_run_ms_worker():
    """Branch and advance trajectory through ms_worker."""
    fmodel = SimpleFModel
    parameters = {"trajectory" : {"end_time": 0.01,
                                  "step_size": 0.001,
                                  "targetscore": 0.25}}
    t_test = Trajectory(fmodel, parameters, 1)
    t_test.advance()
    b_test = ms_worker(0.01, t_test,
                       2, 0.049,
                       time.monotonic() + 10.0,
                       False, "testDB")
    assert b_test.id() == 2
    assert isclose(b_test.scoreMax(), 0.1, abs_tol=1e-9)
    assert b_test.isConverged() is False

def test_run_ms_worker_outoftime(caplog : pytest.LogCaptureFixture):
    """Advance trajectory through pool_worker running out of time."""
    fmodel = DoubleWellModel
    parameters = {"trajectory" : {"end_time": 10.0,
                                  "step_size": 0.01,
                                  "targetscore": 0.75},
                  "tams": {"loglevel": "DEBUG"},
                  "model": {"slow_factor": 0.003}}
    setup_logger(parameters)
    t_test = Trajectory(fmodel, parameters, 1)
    t_test.advance()
    _ = ms_worker(10.0, t_test,
                  2, 0.1,
                  time.monotonic() + 0.1,
                  False, "testDB")
    assert "advance ran out of time" in caplog.text

def test_run_ms_worker_advanceerror():
    """Advance trajectory through pool_worker running into error."""
    fmodel = FailingFModel
    parameters = {"trajectory" : {"end_time": 1.0,
                                  "step_size": 0.001,
                                  "targetscore": 0.75},
                  "tams": {"loglevel": "DEBUG"}}
    setup_logger(parameters)
    t_test = Trajectory(fmodel, parameters, 1)
    t_test.advance(0.01)
    with pytest.raises(RuntimeError):
        _ = ms_worker(10.0, t_test,
                      5, 0.04,
                      time.monotonic() + 1.0,
                      False, "testDB")
