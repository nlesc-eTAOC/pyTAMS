"""Tests for the pytams.xmlutils class."""
import pytest
import pytams.xmlutils as pxml


def test_castTypes():
    """Test manual type casting of XML elements."""
    elem = pxml.new_element("test", 1)
    castedElem = pxml.manualCast(elem)
    assert isinstance(castedElem[1], int)
    elem = pxml.new_element("test", 1.0)
    castedElem = pxml.manualCast(elem)
    assert isinstance(castedElem[1], float)
    elem = pxml.new_element("test", complex(2, 1))
    castedElem = pxml.manualCast(elem)
    assert isinstance(castedElem[1], complex)
    elem = pxml.new_element("test", True)
    castedElem = pxml.manualCast(elem)
    assert isinstance(castedElem[1], bool)
    elem = pxml.new_element("test", "test")
    castedElem = pxml.manualCast(elem)
    assert isinstance(castedElem[1], str)
    elem = pxml.new_element("test", [1, 1])
    with pytest.raises(Exception):
        castedElem = pxml.manualCast(elem)


def test_castSnapshot():
    """Test casting of XML trajectory snapshot."""
    snap = pxml.make_xml_snapshot(1, 0.0, 1.0, 0.0, 10.0)
    time, score, noise, state = pxml.read_xml_snapshot(snap)
    assert(time == 0.0)
    assert(score == 1.0)
    assert(noise == 0.0)
    assert(state == 10.0)
