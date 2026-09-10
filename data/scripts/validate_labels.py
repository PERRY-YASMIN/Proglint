"""data/scripts/validate_labels.py - Validate bounding box normalization and class IDs."""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] pathlib.Path for filesystem paths
from pathlib import Path

# ==============================================================================
# 2. LABEL SYNTAX & BOUND CHECKING
# ==============================================================================
# [DEF: Labels Directory]
labels_dir = Path("data/processed/labels")
label_files = list(labels_dir.glob("*.txt"))

invalid_files = []

# [FUNCTION: Integrity Scan Loop] Checks class==0 and [0, 1] coordinate bounds
for label_file in label_files:
    with open(label_file, "r") as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):
        values = line.strip().split()

        if len(values) != 5:
            invalid_files.append(
                f"{label_file.name}, line {line_number}: wrong number of values"
            )
            continue

        class_id, x, y, width, height = map(float, values)

        # Class ID must be 0 (person)
        if class_id != 0:
            invalid_files.append(
                f"{label_file.name}, line {line_number}: invalid class"
            )

        # Normalized coordinates must be within [0, 1]
        if not (0 <= x <= 1):
            invalid_files.append(
                f"{label_file.name}, line {line_number}: invalid x"
            )

        if not (0 <= y <= 1):
            invalid_files.append(
                f"{label_file.name}, line {line_number}: invalid y"
            )

        if not (0 < width <= 1):
            invalid_files.append(
                f"{label_file.name}, line {line_number}: invalid width"
            )

        if not (0 < height <= 1):
            invalid_files.append(
                f"{label_file.name}, line {line_number}: invalid height"
            )

print("Label files checked:", len(label_files))
print("Invalid entries:", len(invalid_files))

if invalid_files:
    print("\n".join(invalid_files))
    print("Label validation: FAIL")
else:
    print("Label validation: PASS")
