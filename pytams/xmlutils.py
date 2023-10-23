import xml.etree.cElementTree as ET


def dict_to_xml(tag: str, d):
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
