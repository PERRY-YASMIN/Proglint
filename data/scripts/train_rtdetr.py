"""data/scripts/train_rtdetr.py - Fine-tune RT-DETR-L on PETS-2009 CCTV Dataset.

Pipeline:
1. Loads pre-trained base model weights (rtdetr-l.pt)
2. Fine-tunes on PETS-2009 access-control benchmark for 15 epochs on CUDA device 0
3. Automatically copies best checkpoint to runs/custom_train/best_rtdetr.pt
"""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] os, shutil for copying checkpoints
import os, shutil
# [FROM: PyTorch] torch for CUDA GPU detection
import torch
# [FROM: ultralytics] RTDETR Vision Transformer class
from ultralytics import RTDETR

# ==============================================================================
# 2. HARDWARE & DATASET CONFIGURATION
# ==============================================================================
# [INIT: Device Detection] Checks for NVIDIA RTX 4060 GPU
dev = 0 if torch.cuda.is_available() else "cpu"

# [DEF: yaml_path] Resolves dataset configuration manifest
yaml_path = "data/processed/pets2009.yaml" if os.path.exists("data/processed/pets2009.yaml") else "data/processed/pets.yaml"

print(f"[RT-DETR] Initializing 15-epoch fine-tuning on device: {dev}")

# ==============================================================================
# 3. MODEL INITIALIZATION & TRAINING EXECUTION
# ==============================================================================
# [INIT: Model Load] Initializes base RT-DETR model from rtdetr-l.pt
# [FROM: ultralytics.RTDETR]
model = RTDETR("rtdetr-l.pt")

# [FUNCTION: Launch Training]
# Runs 15-epoch transfer learning without modifying internal transformer neural architectures
results = model.train(
    data=yaml_path,        # Dataset manifest path
    epochs=15,             # 15 epochs for convergence
    imgsz=640,             # Standard 640x640 resolution
    batch=8,               # Batch size 8 fits in 8GB VRAM
    device=dev,            # GPU 0
    workers=0,             # workers=0 avoids Windows multiprocessing deadlock
    project="runs/detect", # Output directory
    name="train-3",        # Run name
    exist_ok=True
)

# ==============================================================================
# 4. CHECKPOINT CONSOLIDATION
# ==============================================================================
# [FUNCTION: Save Best Checkpoint] Copies best.pt to standard project weight path
best_src = os.path.join("runs", "detect", "train-3", "weights", "best.pt")
custom_dst = os.path.join("runs", "custom_train", "best_rtdetr.pt")
if os.path.exists(best_src):
    os.makedirs(os.path.dirname(custom_dst), exist_ok=True)
    shutil.copy2(best_src, custom_dst)
    print(f"[RT-DETR] Saved fine-tuned checkpoint to: {custom_dst}")
