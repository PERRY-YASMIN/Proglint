"""data/scripts/convert_annotations.py - Batch convert PETS-2009 XMLs to YOLO text labels."""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] xml.etree.ElementTree for parsing XML, pathlib.Path for filesystem paths
import xml.etree.ElementTree as ET
from pathlib import Path

# ==============================================================================
# 2. PATHS & IMAGE CONSTANTS
# ==============================================================================
# [DEF & INIT: Paths] Input raw directory, output processed image and label directories
RAW_DIR = Path("data/raw")
IMAGE_DIR = Path("data/processed/images")
LABEL_DIR = Path("data/processed/labels")

# [DEF: Image Resolution] Standard PETS-2009 CCTV frame resolution (768x576)
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 576

# [FUNCTION: Ensure Directories Exist]
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 3. XML PARSING & YOLO FORMAT CONVERSION
# ==============================================================================
total_frames = 0

# [FUNCTION: XML Ingestion Loop] Iterates through raw CVML XML files
for xml_path in sorted(RAW_DIR.glob("*.xml")):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence = xml_path.stem.replace("PETS2009-", "")
    frames = root.findall("frame")

    for frame in frames:
        frame_number = int(frame.get("number"))
        label_path = LABEL_DIR / f"{sequence}_frame_{frame_number:04d}.txt"

        # [FUNCTION: Normalize Coordinates] Converts pixel bounding boxes to [0, 1] relative YOLO format
        with open(label_path, "w") as file:
            for obj in frame.findall("objectlist/object"):
                box = obj.find("box")
                xc = float(box.get("xc"))
                yc = float(box.get("yc"))
                width = float(box.get("w"))
                height = float(box.get("h"))

                x_center = xc / IMAGE_WIDTH
                y_center = yc / IMAGE_HEIGHT
                norm_width = width / IMAGE_WIDTH
                norm_height = height / IMAGE_HEIGHT

                # Class 0 = person
                file.write(
                    f"0 {x_center:.6f} {y_center:.6f} "
                    f"{norm_width:.6f} {norm_height:.6f}\n"
                )

        total_frames += 1

    print(f"{sequence}: {len(frames)} frames converted")

print(f"\nTotal frames converted: {total_frames}")
print(f"Labels saved to: {LABEL_DIR}")
