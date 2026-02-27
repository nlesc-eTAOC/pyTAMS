"""Tests for the pytams.bin functions."""
from pathlib import Path
import pytest
import toml
from pytams.bin import tams_alive
from pytams.bin import tams_template_model
from pytams.bin import tams_run


def test_tams_alive(capsys: pytest.CaptureFixture[str]):
    """Test TAMS check function."""
    tams_alive()
    assert "rare-event finder tool" in capsys.readouterr().out

def test_tams_template_model():
    """Test TAMS new model init function."""
    tams_template_model(a_args=[])
    assert Path("./MyNewClass.py").exists()
    Path("./MyNewClass.py").unlink(missing_ok=True)

def test_tams_template_model_with_name():
    """Test TAMS new model init function."""
    tams_template_model(a_args=["-n", "MyCustomClass"])
    assert Path("./MyCustomClass.py").exists()
    Path("./MyCustomClass.py").unlink(missing_ok=True)

def test_tams_run():
    """Test TAMS run."""
    params_dict = {
        "tams": {"ntrajectories": 20, "nsplititer": 20, "walltime": 20.0, "deterministic": True},
        "runner": {"type": "asyncio", "nworker_init": 1, "nworker_iter": 1},
        "trajectory": {"end_time": 6.0, "step_size": 0.01, "targetscore": 0.6},
        "model": {"slow_factor": 0.00000001, "noise_amplitude": 0.6},
    }
    with Path("input.toml").open("w") as f:
        toml.dump(params_dict, f)
    tams_run(a_args=["-m", "./tests/dwmodel.py",
                     "-i", "input.toml"])
    Path("input.toml").unlink(missing_ok=True)

def test_tams_run_fail_two_fmodel():
    """Test TAMS run."""
    with pytest.raises(RuntimeError):
        tams_run(a_args=["-m", "./tests/models.py",
                         "-i", "input.toml"])

def test_tams_run_fail_nofmodel():
    """Test TAMS run."""
    with pytest.raises(RuntimeError):
        tams_run(a_args=["-m", "./tests/test_xmlutils.py",
                         "-i", "input.toml"])
