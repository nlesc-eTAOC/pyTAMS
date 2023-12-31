import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np


class XMLUtilsError(Exception):
    """Exception class for the xmlutils."""

    pass


def manualCast(elem: ET.Element):
    """Manually cast XML elements reads."""
    if elem.attrib["type"] == "int":
        return elem.tag, int(elem.text)
    elif elem.attrib["type"] == "float":
        return elem.tag, float(elem.text)
    elif elem.attrib["type"] == "float64":
        return elem.tag, np.float64(elem.text)
    elif elem.attrib["type"] == "complex":
        return elem.tag, complex(elem.text)
    elif elem.attrib["type"] == "bool":
        if (elem.text == "True"):
            return elem.tag, True
        else:
            return elem.tag, False
    elif elem.attrib["type"] == "str":
        return elem.tag, str(elem.text)
    elif elem.attrib["type"] == "ndarray":
        stripped_text = elem.text.replace("[", "").replace("]", "").replace("  ", " ")
        return elem.tag, np.fromstring(stripped_text, sep=" ")
    elif elem.attrib["type"] == "datetime":
        return elem.tag, datetime.strptime(elem.text, "%Y-%m-%d %H:%M:%S.%f")
    else:
        raise XMLUtilsError(
            "Type {} not handled by manualCast !".format(elem.attrib["type"])
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
        tag, entry = manualCast(child)
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


def make_xml_snapshot(idx: int, time: float, score: float, state) -> ET.Element:
    """Return a snapshot in XML elemt format.

    Args:
        idx: snapshot index
        time: the time stamp
        score: the snapshot score function
        state: the associated state
    """
    elem = ET.Element("Snap_{:07d}".format(idx))
    elem.attrib["time"] = str(time)
    elem.attrib["score"] = str(score)
    elem.attrib["type"] = type(state).__name__
    elem.text = str(state)

    return elem


def read_xml_snapshot(snap: ET.Element):
    """Return snapshot data from an XML snapshot elemt.

    Args:
        snap: an XML snapshot elemt
    """
    time = float(snap.attrib["time"])
    score = float(snap.attrib["score"])
    _, state = manualCast(snap)

    return time, score, state
