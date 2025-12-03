"""Tests for the pytams.bin functions."""
from pathlib import Path
import pytest
from pytams.bin import tams_alive
from pytams.bin import tams_template_model


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

