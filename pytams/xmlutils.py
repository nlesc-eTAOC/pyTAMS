import xml.etree.ElementTree as ET


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
        child.text = str(val)
        elem.append(child)

    return elem


def xml_to_dict(elem: ET.Element):
    """Return an dictionnary an Element.

    Args:
        tag: a root tag
        elem: an etree element
    """
    d = {}
    for child in elem:
        # Append to dict
        d[child.tag] = eval(child.text)

    return d


def new_element(key: str, text):
    """Return an Element from two args.

    Args:
        key: the element key
        text: the element value
    """
    elem = ET.Element(key)
    elem.text = str(text)

    return elem


def make_xml_snapshot(idx: int, time: float, score: float, state):
    """Return a snapshot in XML elemt format.

    Args:
        idx: snapshot index
        time: the time stamp
        score: the snapshot score function
        state: the associated state
    """
    elem = ET.Element("Snap_{:06d}".format(idx))
    elem.attrib["time"] = str(time)
    elem.attrib["score"] = str(score)
    elem.attrib["state"] = str(state)

    return elem


def read_xml_snapshot(snap: ET.Element):
    """Return snapshot data from an XML snapshot elemt.

    Args:
        snap: an XML snapshot elemt
    """
    time = eval(snap.attrib["time"])
    score = eval(snap.attrib["score"])
    state = eval(snap.attrib["state"])

    return time, score, state
