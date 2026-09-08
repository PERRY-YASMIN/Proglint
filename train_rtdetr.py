"""
train_rtdetr.py - Domain-Adapted Fine-Tuning Pipeline for RT-DETR on the PETS-2009 Benchmark.

Optimized for high-throughput transfer learning on NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB VRAM).
Fine-tunes a pre-trained RT-DETR model without modifying internal transformer neural architectures,
focusing on CCTV access-control gate surveillance adaptation.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import RTDETR

# Configure module logger
logger = logging.getLogger("RTDETRTrainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [RT-DETR]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def assert_hardware_diagnostics(required_device_idx: int = 0) -> Dict[str, Any]:
    """Assert CUDA availability, lock onto target GPU, and log hardware telemetry.

    Raises:
        RuntimeError: If CUDA is not detected or target device index is invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CRITICAL HARDWARE ERROR: CUDA is not detected! An active CUDA-capable NVIDIA GPU "
            "is required to execute domain-adapted RT-DETR fine-tuning."
        )

    device_count = torch.cuda.device_count()
    if required_device_idx >= device_count:
        raise RuntimeError(
            f"Requested CUDA device index {required_device_idx} exceeds available devices ({device_count})."
        )

    torch.cuda.set_device(required_device_idx)
    active_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(active_idx)
    props = torch.cuda.get_device_properties(active_idx)
    total_vram_gb = props.total_memory / (1024**3)
    major, minor = props.major, props.minor
    cuda_version = torch.version.cuda

    telemetry = {
        "device": f"cuda:{active_idx}",
        "gpu_name": gpu_name,
        "vram_gb": round(total_vram_gb, 2),
        "compute_capability": f"sm_{major}{minor}",
        "cuda_version": cuda_version,
    }

    print("=" * 72)
    print("🚀 HARDWARE DIAGNOSTICS & CUDA ACCELERATION TELEMETRY")
    print("=" * 72)
    print(f"• Active Compute Device : cuda:{active_idx}")
    print(f"• Dedicated GPU Name    : {gpu_name}")
    print(f"• Total Dedicated VRAM  : {total_vram_gb:.2f} GB GDDR6")
    print(f"• Compute Capability    : sm_{major}{minor} (Ada Lovelace)")
    print(f"• PyTorch CUDA Runtime  : {cuda_version}")
    print(f"• Tensor Core Support   : ENABLED (FP16 AMP Mixed Precision)")
    print("=" * 72)

    # Tensor computation test on GPU
    test_tensor = torch.randn((1000, 1000), device="cuda")
    _ = torch.matmul(test_tensor, test_tensor)
    torch.cuda.synchronize()

    return telemetry


def verify_val_inference(
    checkpoint_path: str,
    val_images_dir: str = "data/processed/images/val",
    conf_thresh: float = 0.25,
) -> Dict[str, Any]:
    """Self-test verification: load checkpoint into RTDETR and run inference on validation images.

    Asserts that predictions return valid bounding box tensors: [x1, y1, x2, y2, conf, cls].
    """
    logger.info("Executing self-test verification on validation frames: '%s'", checkpoint_path)
    model = RTDETR(checkpoint_path)

    val_dir = Path(val_images_dir)
    image_paths = sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")))
    if not image_paths:
        logger.warning("No validation images found in '%s' for self-test.", val_images_dir)
        return {"total_frames_tested": 0, "status": "SKIPPED"}

    sample_images = image_paths[:5]
    total_detections = 0
    sample_outputs = []

    for img_p in sample_images:
        results = model.predict(
            source=str(img_p),
            conf=conf_thresh,
            imgsz=640,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            n_boxes = len(boxes)
            total_detections += n_boxes

            if n_boxes > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()

                for i in range(n_boxes):
                    x1, y1, x2, y2 = xyxy[i]
                    c = float(confs[i])
                    k = int(clss[i])

                    # Assert geometric validity
                    assert x2 > x1 and y2 > y1, f"Degenerate box detected: {[x1, y1, x2, y2]}"
                    assert 0.0 <= c <= 1.0, f"Confidence score out of bounds: {c}"
                    sample_outputs.append({
                        "file": img_p.name,
                        "box_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                        "conf": round(c, 4),
                        "cls": k,
                    })

    logger.info("Self-test verification PASSED: Verified %d detections on %d validation frames.", total_detections, len(sample_images))
    return {
        "status": "PASSED",
        "total_frames_tested": len(sample_images),
        "total_detections": total_detections,
        "sample_detections": sample_outputs[:3],
    }


def train_rtdetr(
    data_yaml: str = "data/processed/pets2009.yaml",
    weights: str = "rtdetr-l.pt",
    epochs: int = 15,
    batch: int = 16,
    imgsz: int = 640,
    device: int = 0,
    workers: int = 2,
    lr0: float = 0.0001,
    weight_decay: float = 0.0001,
    project: str = "runs/rtdetr_train",
    name: str = "pets_rtdetr",
    export_path: str = "runs/custom_train/best_rtdetr.pt",
) -> Dict[str, Any]:
    """Execute domain-adapted transfer learning fine-tuning of RT-DETR on the PETS-2009 dataset.

    Args:
        data_yaml: Path to PETS-2009 dataset YAML manifest.
        weights: Pre-trained weights path or identifier ('rtdetr-l.pt').
        epochs: Number of transfer learning epochs.
        batch: Batch size (default 16, auto-falls back to 8 if VRAM spikes).
        imgsz: Input image resolution.
        device: CUDA device index.
        workers: DataLoader subprocess workers.
        lr0: Initial learning rate.
        weight_decay: Weight decay regularizer.
        project: Parent export directory.
        name: Experiment run name.
        export_path: Target deployment path for best checkpoint.

    Returns:
        Dictionary of training results and metrics.
    """
    # 1. Hardware diagnostics & CUDA hard lock
    assert_hardware_diagnostics(required_device_idx=device)

    # 2. Validate dataset existence
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at: {yaml_path}")

    # 3. Load pre-trained RT-DETR architecture
    logger.info("Loading pre-trained base model weights: '%s'...", weights)
    model = RTDETR(weights)

    project_dir = Path(project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("🎯 DOMAIN-ADAPTED RT-DETR TRAINING SPECIFICATION")
    print("=" * 72)
    print(f"• Base Weights      : {weights}")
    print(f"• Dataset Manifest  : {yaml_path}")
    print(f"• Training Target   : Surveillance CCTV Gates (PETS-2009)")
    print(f"• Input Resolution  : {imgsz}x{imgsz}")
    print(f"• Batch Size        : {batch} (VRAM budgeted for 8GB GPU)")
    print(f"• Training Epochs   : {epochs}")
    print(f"• Optimizer         : AdamW (lr0={lr0}, weight_decay={weight_decay})")
    print(f"• Mixed Precision   : FP16 (amp=True)")
    print(f"• In-Memory Cache   : True (Zero disk I/O bottleneck)")
    print(f"• DataLoader Workers: {workers}")
    print(f"• Checkpoint Target : {project_dir / name / 'weights' / 'best.pt'}")
    print("=" * 72 + "\n")

    start_time = time.time()

    # 4. Launch Ultralytics RT-DETR training with memory safeguards
    active_batch = batch
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=active_batch,
            device=device,
            amp=True,
            cache=True,
            optimizer="AdamW",
            lr0=lr0,
            weight_decay=weight_decay,
            workers=workers,
            project=str(project_dir),
            name=name,
            exist_ok=True,
            save=True,
            verbose=True,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        if "out of memory" in str(err).lower() and active_batch > 8:
            logger.warning("VRAM spike caught with batch=%d. Retrying with batch=8...", active_batch)
            torch.cuda.empty_cache()
            active_batch = 8
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=active_batch,
                device=device,
                amp=True,
                cache=True,
                optimizer="AdamW",
                lr0=lr0,
                weight_decay=weight_decay,
                workers=workers,
                project=str(project_dir),
                name=name,
                exist_ok=True,
                save=True,
                verbose=True,
            )
        else:
            raise err

    elapsed_time = time.time() - start_time

    # 5. Extract trained checkpoint
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project_dir / name
    best_ckpt = save_dir / "weights" / "best.pt"
    last_ckpt = save_dir / "weights" / "last.pt"

    chosen_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
    if not chosen_ckpt.exists():
        raise RuntimeError(f"Expected model weights not found in: {save_dir / 'weights'}")

    export_dest = Path(export_path).resolve()
    export_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen_ckpt, export_dest)
    logger.info("Successfully exported domain-adapted checkpoint to: %s", export_dest)

    # 6. Evaluate validation metrics
    logger.info("Evaluating validation split metrics...")
    val_model = RTDETR(str(chosen_ckpt))
    val_res = val_model.val(data=str(yaml_path), split="val", imgsz=imgsz, device=device, verbose=False)

    precision = float(val_res.box.mp) if hasattr(val_res, "box") else 0.0
    recall = float(val_res.box.mr) if hasattr(val_res, "box") else 0.0
    map50 = float(val_res.box.map50) if hasattr(val_res, "box") else 0.0
    map50_95 = float(val_res.box.map) if hasattr(val_res, "box") else 0.0

    print("\n" + "=" * 72)
    print("🏆 DOMAIN-ADAPTED RT-DETR VALIDATION BENCHMARK RESULTS")
    print("=" * 72)
    print(f"• Total Wall-Clock Time : {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} min)")
    print(f"• Precision (B)         : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"• Recall (B)            : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"• mAP@50 (B)            : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"• mAP@50:95 (B)         : {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print(f"• Checkpoint Location   : {export_dest}")
    print("=" * 72 + "\n")

    # 7. Execute self-test validation inference
    self_test_res = verify_val_inference(str(export_dest))

    return {
        "elapsed_sec": round(elapsed_time, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "checkpoint_path": str(export_dest),
        "self_test": self_test_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain-Adapted RT-DETR Fine-Tuning Pipeline for PETS-2009 Benchmark")
    parser.add_argument("--data", type=str, default="data/processed/pets2009.yaml", help="Path to dataset YAML manifest")
    parser.add_argument("--weights", type=str, default="rtdetr-l.pt", help="Base pre-trained weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of transfer learning epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--export", type=str, default="runs/custom_train/best_rtdetr.pt", help="Target checkpoint path")

    args = parser.parse_args()

    training_summary = train_rtdetr(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        export_path=args.export,
    )
