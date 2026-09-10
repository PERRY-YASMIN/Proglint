"""data/scripts/inspect_annotations.py - Quick inspect raw PETS-2009 CVML XML attributes."""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] xml.etree.ElementTree for reading XML nodes
import xml.etree.ElementTree as ET

# ==============================================================================
# 2. XML INSPECTION
# ==============================================================================
# [DEF: Raw XML Path]
xml_path = "data/raw/PETS2009-S2L1.xml"

try:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame = root.find("frame")
    if frame is not None:
        print("Frame number:", frame.get("number"))
        objects = frame.findall("objectlist/object")
        print("Number of objects:", len(objects))

        for obj in objects:
            box = obj.find("box")
            if box is not None:
                print(
                    "Object ID:", obj.get("id"),
                    "| xc:", box.get("xc"),
                    "| yc:", box.get("yc"),
                    "| width:", box.get("w"),
                    "| height:", box.get("h")
                )
except FileNotFoundError:
    print(f"File not found: {xml_path}")
