"""Tests for the pytams.xmlutils class."""
import numpy as np
import pytest
import pytams.xmlutils as pxml
import datetime


def test_castTypes():
    """Test manual type casting of XML elements."""
    elem = pxml.new_element("test", 1)
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], int)
    elem = pxml.new_element("test", 1.0)
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], float)
    elem = pxml.new_element("test", complex(2, 1))
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], complex)
    elem = pxml.new_element("test", True)
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], bool)
    elem = pxml.new_element("test", "test")
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], str)
    elem = pxml.new_element("test", np.ones(2))
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], np.ndarray)
    elem = pxml.new_element("test", np.ones(5, dtype=int))
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], np.ndarray)
    elem = pxml.new_element("test", np.ones(5,dtype=bool))
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], np.ndarray)
    elem = pxml.new_element("test", {'key1': 'val1','key2': "val2", 'key3': 1, 'key4': 1.0})
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], dict)
    elem = pxml.new_element("test", datetime.datetime.now(tz=datetime.timezone.utc))
    casted_elem = pxml.manual_cast(elem)
    assert isinstance(casted_elem[1], datetime.datetime)
    elem = pxml.new_element("test", [1, 1])
    with pytest.raises(Exception):
        casted_elem = pxml.manual_cast(elem)


def test_castSnapshot():
    """Test casting of XML trajectory snapshot."""
    snap = pxml.make_xml_snapshot(1, 0.0, 1.0, 0.0, 10.0)
    time, score, noise, state = pxml.read_xml_snapshot(snap)
    assert(time == 0.0)
    assert(score == 1.0)
    assert(noise == 0.0)
    assert(state == 10.0)
