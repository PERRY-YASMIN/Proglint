"""data/scripts/check_ultralytics_dataset.py - Validate dataset manifest using Ultralytics validator."""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: ultralytics] check_det_dataset to verify YAML paths and class dictionaries
from ultralytics.data.utils import check_det_dataset

# ==============================================================================
# 2. VALIDATION CHECK
# ==============================================================================
# [FUNCTION: Run Dataset Check]
data = check_det_dataset("data/processed/pets.yaml")

print("Dataset check: PASS")
print("Train path:", data["train"])
print("Validation path:", data["val"])
print("Classes:", data["names"])
