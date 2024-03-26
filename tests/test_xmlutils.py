"""Tests for the pytams.xmlutils class."""
import pytest
import pytams.xmlutils as pxml


def test_castTypes():
    """Test manual type casting of XML elements."""
    elem = pxml.new_element("test", 1)
    castedElem = pxml.manualCastElem(elem)
    assert isinstance(castedElem[1], int)
    elem = pxml.new_element("test", 1.0)
    castedElem = pxml.manualCastElem(elem)
    assert isinstance(castedElem[1], float)
    elem = pxml.new_element("test", complex(2, 1))
    castedElem = pxml.manualCastElem(elem)
    assert isinstance(castedElem[1], complex)
    elem = pxml.new_element("test", True)
    castedElem = pxml.manualCastElem(elem)
    assert isinstance(castedElem[1], bool)
    elem = pxml.new_element("test", "test")
    castedElem = pxml.manualCastElem(elem)
    assert isinstance(castedElem[1], str)
    elem = pxml.new_element("test", [1, 1])
    with pytest.raises(Exception):
        castedElem = pxml.manualCastElem(elem)
