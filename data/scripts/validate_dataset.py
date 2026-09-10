"""data/scripts/validate_dataset.py - Verifies 1-to-1 pairing of images and labels."""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] pathlib.Path for path operations
from pathlib import Path

# ==============================================================================
# 2. PAIRING VALIDATION LOGIC
# ==============================================================================
# [DEF: Paths] Target image and label folders
images_dir = Path("data/processed/images")
labels_dir = Path("data/processed/labels")

# [FUNCTION: Set Difference Analysis] Finds missing counterparts
images = {file.stem for file in images_dir.glob("*.jpg")}
labels = {file.stem for file in labels_dir.glob("*.txt")}

missing_labels = images - labels
missing_images = labels - images

print("Images:", len(images))
print("Labels:", len(labels))
print("Missing labels:", len(missing_labels))
print("Missing images:", len(missing_images))

if missing_labels:
    print("Images without labels:", sorted(missing_labels))

if missing_images:
    print("Labels without images:", sorted(missing_images))

if not missing_labels and not missing_images:
    print("Dataset pairing: PASS")
