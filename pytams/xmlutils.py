import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np


class XMLUtilsError(Exception):
    """Exception class for the xmlutils."""

    pass


def manualCastElem(elem: ET.Element):
    """Manually cast XML elements reads."""
    return elem.tag, manualCast(elem.attrib["state_type"], elem.text)


def manualCastNoise(elem: ET.Element):
    """Manually cast XML elements noise data."""
    return elem.tag, manualCast(elem.attrib["noise_type"], elem.attrib["noise"])


def manualCast(type_str: str,
               elem_text: str):
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
            True
        else:
            False
    elif type_str == "str":
        return str(elem_text)
    elif type_str == "ndarray":
        stripped_text = elem_text.replace("[", "").replace("]", "").replace("  ", " ")
        return np.fromstring(stripped_text, sep=" ")
    elif type_str == "datetime":
        return datetime.strptime(elem_text, "%Y-%m-%d %H:%M:%S.%f")
    else:
        raise XMLUtilsError(
            "Type {} not handled by manualCast !".format(type_str)
        )


def dict_to_xml(tag: str, d) -> ET.Element:
    """Return an Element from a dictionnary.

    Args:
        tag: a root tag
        d: a dictionary
    """
    elem = ET.Element(tag)
    for key, val in d.items():
        # Append an Element
        child = ET.Element(key)
        child.attrib["type"] = type(val).__name__
        child.text = str(val)
        elem.append(child)

    return elem


def xml_to_dict(elem: ET.Element) -> dict:
    """Return an dictionnary an Element.

    Args:
        tag: a root tag
        elem: an etree element
    """
    d = {}
    for child in elem:
        tag, entry = manualCastElem(child)
        d[tag] = entry

    return d


def new_element(key: str, val) -> ET.Element:
    """Return an Element from two args.

    Args:
        key: the element key
        val: the element value
    """
    elem = ET.Element(key)
    elem.attrib["type"] = type(val).__name__
    elem.text = str(val)

    return elem


def make_xml_snapshot(idx: int,
                      time: float,
                      score: float,
                      noise,
                      state) -> ET.Element:
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
    elem.attrib["noise_type"] = type(noise).__name__
    elem.attrib["noise"] = str(noise)
    elem.attrib["state_type"] = type(state).__name__
    elem.text = str(state)

    return elem


def read_xml_snapshot(snap: ET.Element):
    """Return snapshot data from an XML snapshot elemt.

    Args:
        snap: an XML snapshot elemt
    """
    time = float(snap.attrib["time"])
    score = float(snap.attrib["score"])
    _, noise = manualCastNoise(snap)
    _, state = manualCastElem(snap)

    return time, score, noise, state
