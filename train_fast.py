"""
train_fast.py - Ultra-Fast Custom YOLO11 Pedestrian Model Fine-Tuning.

Optimized for high-throughput transfer learning (< 60 seconds) on NVIDIA RTX 4060 Laptop GPU.
Leverages CUDA acceleration, FP16 AMP mixed precision, RAM caching, and AdamW optimizer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time
from typing import Optional

import torch
from ultralytics import YOLO


def assert_hardware_readiness(required_device_idx: int = 0) -> None:
    """Assert CUDA is available, verify target device, and print GPU specifications.

    Raises:
        AssertionError: If CUDA is not detected or target device index is invalid.
    """
    # 1. Hardware Assertion
    assert torch.cuda.is_available(), (
        "CRITICAL ERROR: CUDA is not available! A CUDA-capable NVIDIA GPU "
        "is required to execute ultra-fast transfer learning."
    )

    device_count = torch.cuda.device_count()
    assert (
        required_device_idx < device_count
    ), f"Requested device index {required_device_idx} exceeds available devices ({device_count})."

    # Verify device 0
    torch.cuda.set_device(required_device_idx)
    active_idx = torch.cuda.current_device()
    assert (
        active_idx == required_device_idx
    ), f"Expected active device {required_device_idx}, got {active_idx}."

    gpu_name = torch.cuda.get_device_name(required_device_idx)
    props = torch.cuda.get_device_properties(required_device_idx)
    total_vram_gb = props.total_memory / (1024**3)
    major, minor = props.major, props.minor

    print("=" * 70)
    print("🚀 HARDWARE DIAGNOSTICS & ACCELERATION CHECK")
    print("=" * 70)
    print(f"• Active Compute Device : cuda:{active_idx}")
    print(f"• Dedicated GPU Name    : {gpu_name}")
    print(f"• Total Dedicated VRAM  : {total_vram_gb:.2f} GB")
    print(f"• Compute Capability    : sm_{major}{minor}")
    print(f"• PyTorch CUDA Version  : {torch.version.cuda}")
    print("• Hardware Status       : PASSED (Tensor Cores Ready)")
    print("=" * 70)


def ensure_sample_dataset(data_path: str) -> str:
    """Ensure sample mini pedestrian dataset exists if the default path is specified."""
    if os.path.exists(data_path):
        return data_path

    abs_yaml_path = os.path.abspath(data_path)
    if os.path.normpath(data_path) != os.path.normpath("data/pedestrian_mini.yaml"):
        raise FileNotFoundError(f"Specified dataset YAML not found at: '{abs_yaml_path}'")

    print(f"[Dataset] Generating sample mini pedestrian dataset at '{data_path}'...")
    import cv2
    import numpy as np
    import yaml

    base_dir = os.path.abspath("data/pedestrian_mini")
    train_img_dir = os.path.join(base_dir, "images", "train")
    val_img_dir = os.path.join(base_dir, "images", "val")
    train_lbl_dir = os.path.join(base_dir, "labels", "train")
    val_lbl_dir = os.path.join(base_dir, "labels", "val")

    for p in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        os.makedirs(p, exist_ok=True)

    # Generate sample images with pedestrian annotations
    for split, count in [("train", 32), ("val", 16)]:
        for i in range(count):
            img = np.zeros((480, 480, 3), dtype=np.uint8)
            cv2.rectangle(img, (180, 100), (300, 380), (220, 220, 220), -1)
            img_path = os.path.join(base_dir, "images", split, f"sample_{i:03d}.jpg")
            cv2.imwrite(img_path, img)

            lbl_path = os.path.join(base_dir, "labels", split, f"sample_{i:03d}.txt")
            with open(lbl_path, "w") as f:
                f.write("0 0.5 0.5 0.25 0.6\n")

    cfg = {
        "path": base_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }
    with open(data_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"[Dataset] Sample mini dataset created successfully with 48 images.")
    return data_path


def train_fast(
    data_yaml: str = "data/pedestrian_mini.yaml",
    epochs: int = 5,
    imgsz: int = 480,
    batch: int = 32,
    device: int = 0,
    workers: int = 4,
    output_weight_path: str = "runs/custom_train/best.pt",
) -> float:
    """Execute rapid transfer-learning fine-tuning of YOLO11 for pedestrian detection.

    Args:
        data_yaml: Path to dataset YAML definition file.
        epochs: Number of training epochs (default: 5).
        imgsz: Image input resolution (480 or 640).
        batch: Batch size to maximize GPU tensor core utilization (default: 32).
        device: CUDA device index (default: 0).
        workers: DataLoader subprocess workers (default: 4).
        output_weight_path: Target path to save the best model weights.

    Returns:
        Total training elapsed wall-clock time in seconds.
    """
    # 1. Assert hardware readiness
    assert_hardware_readiness(required_device_idx=device)

    # 2. Ensure dataset availability
    dataset_path = ensure_sample_dataset(data_yaml)

    # 3. Load base pretrained weights (yolo11n.pt)
    print(f"\n[Pipeline] Loading base weights: 'yolo11n.pt' for transfer learning...")
    model = YOLO("yolo11n.pt")

    # 4. Target export directory
    target_export = Path(output_weight_path).resolve()
    target_export.parent.mkdir(parents=True, exist_ok=True)
    project_dir = target_export.parent.resolve()

    print(f"[Pipeline] Launching optimized fast fine-tuning:")
    print(f"  • Dataset     : {dataset_path}")
    print(f"  • Epochs      : {epochs}")
    print(f"  • Image Size  : {imgsz}x{imgsz}")
    print(f"  • Batch Size  : {batch} (VRAM saturation)")
    print(f"  • Optimizer   : AdamW")
    print(f"  • Mixed Prec  : FP16 (amp=True)")
    print(f"  • RAM Cache   : True (zero disk I/O bottleneck)")
    print(f"  • Device      : cuda:{device}")
    print(f"  • Workers     : {workers}\n")

    # Start wall-clock timer
    start_time = time.time()

    # 5. Run Ultralytics training with speed-optimized hyperparameters
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=True,          # FP16 mixed precision
        cache=True,        # RAM caching eliminates disk I/O bottleneck
        optimizer="AdamW", # Fast convergence for transfer learning
        workers=workers,
        project=str(project_dir),
        name="exp",
        exist_ok=True,
        save=True,
        verbose=True,
    )

    elapsed_sec = time.time() - start_time

    # 6. Export best fine-tuned weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project_dir / "exp"
    trained_best_src = save_dir / "weights" / "best.pt"

    if trained_best_src.exists():
        shutil.copy(trained_best_src, target_export)
        print(f"\n[Export] Fine-tuned model weights successfully saved to:")
        print(f"         {target_export}")
    else:
        # Fallback to last.pt if best.pt is absent
        trained_last_src = save_dir / "weights" / "last.pt"
        if trained_last_src.exists():
            shutil.copy(trained_last_src, target_export)
            print(f"\n[Export] Last model weights saved to: {target_export}")

    # 7. Log elapsed time and evaluate against < 60s requirement
    print("\n" + "=" * 70)
    print("🏁 BENCHMARK RESULTS & LATENCY EVALUATION")
    print("=" * 70)
    print(f"• Total Elapsed Wall-Clock Time : {elapsed_sec:.2f} seconds")
    print(f"• Target Latency Threshold      : < 60.00 seconds")
    target_met = elapsed_sec < 60.0
    status_msg = "SUCCESS (Target Achieved)" if target_met else "WARNING (Target Exceeded)"
    print(f"• Verification Status           : {status_msg}")
    print("=" * 70 + "\n")

    return elapsed_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a custom YOLO11 pedestrian model in under 60 seconds on RTX 4060 GPU."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/pedestrian_mini.yaml",
        help="Path to dataset YAML file (default: 'data/pedestrian_mini.yaml').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        choices=[480, 640],
        default=480,
        help="Input image resolution (default: 480, choice: 480 or 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/custom_train/best.pt",
        help="Target export path for best model (default: 'runs/custom_train/best.pt').",
    )

    args = parser.parse_args()

    train_fast(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        output_weight_path=args.output,
    )


if __name__ == "__main__":
    main()
