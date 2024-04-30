import ast
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
import numpy as np


class XMLUtilsError(Exception):
    """Exception class for the xmlutils."""

    pass


def manualCastSnap(elem: ET.Element) -> Any:
    """Manually cast XML snapshot state."""
    return elem.tag, manualCastStr(elem.attrib["state_type"], elem.text)


def manualCastSnapNoise(elem: ET.Element) -> Any:
    """Manually cast XML snapshot noise."""
    return elem.tag, manualCastStr(elem.attrib["noise_type"], elem.attrib["noise"])


def manualCast(elem: ET.Element) -> Any:
    """Manually cast XML elements reads."""
    return elem.tag, manualCastStr(elem.attrib["type"], elem.text)


def manualCastStr(type_str: str,
                  elem_text: str) -> Any:
    """Manually cast from strings."""
    if type_str == "int":
        return int(elem_text)
    elif type_str == "float":
        return float(elem_text)
    elif type_str == "float64":
        return np.float64(elem_text)
    elif type_str == "complex":
        return complex(elem_text)
    elif type_str == "bool":
        if (elem_text == "True"):
            return True
        else:
            return False
    elif (type_str == "str" or type_str == "str_"):
        return str(elem_text)
    elif type_str == "ndarray[float]":
        stripped_text = elem_text.replace("[", "").replace("]", "").replace("  ", " ")
        return np.fromstring(stripped_text, sep=" ")
    elif type_str == "ndarray[int]":
        stripped_text = elem_text.replace("[", "").replace("]", "").replace("  ", " ")
        return np.fromstring(stripped_text, dtype=int, sep=" ")
    elif type_str == "ndarray":     # Default ndarray to float
        stripped_text = elem_text.replace("[", "").replace("]", "").replace("  ", " ")
        return np.fromstring(stripped_text, sep=" ")
    elif type_str == "datetime":
        return datetime.strptime(elem_text, "%Y-%m-%d %H:%M:%S.%f")
    elif type_str == "dict":
        return ast.literal_eval(elem_text)
    else:
        raise XMLUtilsError(
            "Type {} not handled by manualCast !".format(type_str)
        )


def dict_to_xml(tag: str, d: dict) -> ET.Element:
    """Return an Element from a dictionnary.

    Args:
        tag: a root tag
        d: a dictionary
    """
    elem = ET.Element(tag)
    for key, val in d.items():
        # Append an Element
        child = ET.Element(key)
        child.attrib["type"] = get_val_type(val)
        child.text = str(val)
        elem.append(child)

    return elem


def xml_to_dict(elem: ET.Element) -> dict:
    """Return an dictionnary an Element.

    Args:
        elem: an etree element

    Return:
        a dictionary containing the element entries
    """
    d = {}
    for child in elem:
        tag, entry = manualCast(child)
        d[tag] = entry

    return d

def get_val_type(val: Any) -> str:
    """Return the type of val.

    Args:
        val: a value

    Return:
        val type
    """
    base_type = type(val).__name__
    if base_type == "ndarray":
        if val.dtype == "float64":
            base_type = base_type + "[float]"
        elif val.dtype == "int64":
            base_type = base_type + "[int]"
        return base_type
    else:
        return base_type


def new_element(key: str, val: Any) -> ET.Element:
    """Return an Element from two args.

    Args:
        key: the element key
        val: the element value

    Return:
        an ElementTree element
    """
    elem = ET.Element(key)
    elem.attrib["type"] = get_val_type(val)
    elem.text = str(val)

    return elem


def make_xml_snapshot(idx: int,
                      time: float,
                      score: float,
                      noise: Any,
                      state: Any) -> ET.Element:
    """Return a snapshot in XML elemt format.

    Args:
        idx: snapshot index
        time: the time stamp
        score: the snapshot score function
        noise: the stochastic noise
        state: the associated state
    """
    elem = ET.Element("Snap_{:07d}".format(idx))
    elem.attrib["time"] = str(time)
    elem.attrib["score"] = str(score)
    elem.attrib["noise_type"] = get_val_type(noise)
    elem.attrib["noise"] = str(noise)
    elem.attrib["state_type"] = get_val_type(state)
    elem.text = str(state)

    return elem


def read_xml_snapshot(snap: ET.Element):
    """Return snapshot data from an XML snapshot elemt.

    Args:
        snap: an XML snapshot elemt
    """
    time = float(snap.attrib["time"])
    score = float(snap.attrib["score"])
    _, noise = manualCastSnapNoise(snap)
    _, state = manualCastSnap(snap)

    return time, score, noise, state
