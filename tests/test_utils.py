"""Tests for the pytams.utils class."""

import pytest
from pytams.utils import get_module_local_import
from tests.models import SimpleFModel


def test_get_local_import():
    """Test get local import function."""
    mods = get_module_local_import(SimpleFModel.__module__)
    assert len(mods) == 0


def test_get_local_import_fail():
    """Test get local import function with dummy module."""
    with pytest.raises(ValueError):
        _ = get_module_local_import("dummy_module_name")

def test_get_local_import_fail_stdlib():
    """Test get local import function with a module from stdlib."""
    with pytest.raises(FileNotFoundError):
        _ = get_module_local_import("time")
