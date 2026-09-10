"""data/scripts/split_dataset.py - 80/20 train/val split for images and labels."""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] pathlib.Path for filesystem paths, shutil for copying files
from pathlib import Path
import shutil

# ==============================================================================
# 2. DIRECTORY CONFIGURATION
# ==============================================================================
# [DEF: Paths] Source and destination directories for 80/20 train/val split
images_dir = Path("data/processed/images")
labels_dir = Path("data/processed/labels")

train_images = Path("data/processed/train/images")
train_labels = Path("data/processed/train/labels")

val_images = Path("data/processed/val/images")
val_labels = Path("data/processed/val/labels")

# [FUNCTION: Ensure Target Directories Exist]
for directory in [train_images, train_labels, val_images, val_labels]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 3. SPLIT & COPY
# ==============================================================================
# [INIT: File List & 80/20 Ratio Index]
image_files = sorted(images_dir.glob("*.jpg"))
split_index = int(len(image_files) * 0.8)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

# [FUNCTION: Copy Training Split]
for image_file in train_files:
    label_file = labels_dir / f"{image_file.stem}.txt"
    shutil.copy2(image_file, train_images / image_file.name)
    shutil.copy2(label_file, train_labels / label_file.name)

# [FUNCTION: Copy Validation Split]
for image_file in val_files:
    label_file = labels_dir / f"{image_file.stem}.txt"
    shutil.copy2(image_file, val_images / image_file.name)
    shutil.copy2(label_file, val_labels / label_file.name)

print("Total images:", len(image_files))
print("Training images:", len(train_files))
print("Validation images:", len(val_files))
print("Split index:", split_index)
