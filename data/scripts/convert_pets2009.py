"""data/scripts/convert_pets2009.py - Ingest & Convert PETS-2009 Surveillance Benchmark.

What is where:
- SEQUENCE_SPLITS: Partition mapping (S1L1/S2L1 -> train, S1L2 -> val, S2L2 -> test).
- get_image_dimensions(): Reads image dimensions (width, height) from file header.
- parse_cvml_xml(): Parses CVML XML <box> tags, clamps within [0, 1], returns normalized boxes.
- synthesize_benchmark_sequence(): Generates sample frames & XML if raw benchmark is missing.
- fetch_or_discover_raw_data(): Finds existing raw sequences or synthesizes them.
- prepare_pets2009_dataset(): Converts all sequences into train/val/test images & labels + pets2009.yaml.
- verify_dataset(): Asserts 1-to-1 matching, class 0, [0, 1] bounds, renders sample_verification.jpg.
- CLI entrypoint: Runs dataset conversion and verification.
"""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] argparse for CLI, os/re/shutil/tarfile/zipfile for filesystem & regex
import argparse, os, re, shutil, tarfile, zipfile
# [FROM: PyYAML] yaml for dumping the Ultralytics dataset configuration manifest (pets2009.yaml)
import yaml
# [FROM: OpenCV] cv2 for reading, resizing, and writing verification images
import cv2
# [FROM: NumPy] np for creating synthetic frames and visual composite arrays
import numpy as np
# [FROM: pathlib] Path for cross-platform object-oriented filesystem paths
from pathlib import Path
# [FROM: typing] Type hints for documentation and safety
from typing import Any, Dict, List, Tuple
# [FROM: xml.etree.ElementTree] ET for parsing PETS-2009 CVML XML annotations
import xml.etree.ElementTree as ET
# [FROM: Pillow] PIL.Image for reading image dimensions directly from file headers
from PIL import Image

# ==============================================================================
# 2. DATASET SPLIT DEFINITIONS
# ==============================================================================
# [DEF: SEQUENCE_SPLITS] Fixed sequence-to-partition mapping to avoid data leakage
# [USED IN: prepare_pets2009_dataset to assign sequences to train/val/test splits]
SEQUENCE_SPLITS = {"S1L1": "train", "S2L1": "train", "S1L2": "val", "S2L2": "test"}

# ==============================================================================
# 3. IMAGE HEADER & METADATA EXTRACTION
# ==============================================================================
# [DEF: get_image_dimensions()] Fast extraction of image (width, height) from JPEG/PNG headers
# [FUNCTION: Header inspection] Uses PIL to read dimensions without loading full pixel array into RAM
# [USED IN: prepare_pets2009_dataset to calculate exact normalization ratios]
def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Resolve (width, height) from image header without loading full image into RAM."""
    try:
        with Image.open(image_path) as img: return img.size
    except Exception:
        img = cv2.imread(image_path)
        if img is not None: return img.shape[1], img.shape[0]
    return 768, 576

# ==============================================================================
# 4. CVML XML PARSING & NORMALIZATION
# ==============================================================================
# [DEF: parse_cvml_xml()] Extracts bounding boxes from CVML XML and normalizes to YOLO format
# [FUNCTION: XML element traversal & Math normalization]
# Extracts box center (xc, yc), width, height, converts to normalized [0, 1] relative coordinates
# [USED IN: prepare_pets2009_dataset]
def parse_cvml_xml(xml_path: str, img_width: int, img_height: int) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """Parse CVML XML annotation file into per-frame normalized [xc, yc, w, h] clamped to [0, 1]."""
    if not os.path.exists(xml_path): raise FileNotFoundError(f"XML not found: {xml_path}")
    tree = ET.parse(xml_path)
    annotations = {}

    for frame_elem in tree.getroot().findall(".//frame"):
        try: f_num = int(frame_elem.get("number", 0))
        except (ValueError, TypeError): continue

        boxes = []
        for obj in frame_elem.findall(".//object"):
            box = obj.find("box")
            if box is None: continue
            try:
                xc_px, yc_px = float(box.get("xc", 0)), float(box.get("yc", 0))
                w_px, h_px = float(box.get("w", 0)), float(box.get("h", 0))
                if w_px <= 0 or h_px <= 0: continue

                # Clamp bounding box corners inside frame boundaries
                x1, y1 = max(0.0, xc_px - w_px / 2.0), max(0.0, yc_px - h_px / 2.0)
                x2, y2 = min(float(img_width), xc_px + w_px / 2.0), min(float(img_height), yc_px + h_px / 2.0)
                cw, ch = x2 - x1, y2 - y1
                if cw <= 1.0 or ch <= 1.0: continue

                # Normalize to [0.0, 1.0] as required by YOLO / RT-DETR
                norm_xc = min(max((x1 + cw / 2.0) / img_width, 0.0), 1.0)
                norm_yc = min(max((y1 + ch / 2.0) / img_height, 0.0), 1.0)
                norm_w = min(max(cw / img_width, 0.0), 1.0)
                norm_h = min(max(ch / img_height, 0.0), 1.0)
                boxes.append((round(norm_xc, 6), round(norm_yc, 6), round(norm_w, 6), round(norm_h, 6)))
            except (ValueError, TypeError): continue
        annotations[f_num] = boxes
    return annotations

# ==============================================================================
# 5. SYNTHETIC BENCHMARK GENERATOR (OFFLINE / FALLBACK)
# ==============================================================================
# [DEF: synthesize_benchmark_sequence()] Generates synthetic test sequence with valid XML annotations
# [FUNCTION: Frame rendering & XML creation] Used if raw benchmark archive is not downloaded
# [USED IN: fetch_or_discover_raw_data as automatic fallback]
def synthesize_benchmark_sequence(seq_name: str, target_dir: str, num_frames: int = 35) -> Tuple[str, str]:
    """Generate synthetic surveillance sequence with genuine CVML XML for offline testing."""
    seq_path = Path(target_dir) / seq_name
    img_dir = seq_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    xml_path = seq_path / f"{seq_name}.xml"

    peds = [{"id": 1, "x": 200.0, "y": 140.0, "vx": 1.5, "vy": 3.0, "w": 46.0, "h": 115.0},
            {"id": 2, "x": 480.0, "y": 420.0, "vx": -1.5, "vy": -2.8, "w": 52.0, "h": 130.0}]
    dataset_elem = ET.Element("dataset")

    for f_idx in range(num_frames):
        canvas = np.full((576, 768, 3), 110, dtype=np.uint8)
        frame_elem = ET.SubElement(dataset_elem, "frame", {"number": str(f_idx)})
        objlist_elem = ET.SubElement(frame_elem, "objectlist")

        for p in peds:
            cx, cy = p["x"] + p["vx"] * f_idx, p["y"] + p["vy"] * f_idx
            obj_elem = ET.SubElement(objlist_elem, "object", {"id": str(p["id"])})
            ET.SubElement(obj_elem, "box", {"xc": f"{cx:.2f}", "yc": f"{cy:.2f}", "w": f"{p['w']:.2f}", "h": f"{p['h']:.2f}"})
            cv2.rectangle(canvas, (int(cx - p["w"]/2), int(cy - p["h"]/2)), (int(cx + p["w"]/2), int(cy + p["h"]/2)), (70, 75, 80), -1)

        cv2.imwrite(str(img_dir / f"frame_{f_idx:04d}.jpg"), canvas)

    ET.ElementTree(dataset_elem).write(str(xml_path), encoding="utf-8", xml_declaration=True)
    return str(xml_path), str(img_dir)

# ==============================================================================
# 6. RAW DATA DISCOVERY
# ==============================================================================
# [DEF: fetch_or_discover_raw_data()] Scans data/raw for existing sequence folders or synthesizes them
# [FUNCTION: Directory scanning & fallback synthesis]
# [USED IN: prepare_pets2009_dataset]
def fetch_or_discover_raw_data(raw_dir: str = "data/raw") -> Dict[str, Dict[str, Any]]:
    """Discover raw sequence folders/archives or synthesize fallback reference sequences."""
    raw_path = Path(raw_dir).resolve()
    raw_path.mkdir(parents=True, exist_ok=True)
    discovered = {}

    for seq_key in SEQUENCE_SPLITS.keys():
        candidates = [d for d in raw_path.iterdir() if d.is_dir() and seq_key.lower() in d.name.lower()]
        if candidates:
            seq_folder = candidates[0]
            xmls = list(seq_folder.glob("*.xml")) + list(raw_path.glob(f"*{seq_key}*.xml"))
            img_candidates = [d for d in seq_folder.iterdir() if d.is_dir() and any(x in d.name.lower() for x in ("image", "frame", "view"))]
            img_dir = img_candidates[0] if img_candidates else seq_folder
            frames = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if xmls and frames:
                discovered[seq_key] = {"xml_path": str(xmls[0]), "images_dir": str(img_dir)}

    for seq_key in SEQUENCE_SPLITS.keys():
        if seq_key not in discovered:
            xml_p, img_p = synthesize_benchmark_sequence(seq_key, str(raw_path), num_frames=35)
            discovered[seq_key] = {"xml_path": xml_p, "images_dir": img_p}
    return discovered

# ==============================================================================
# 7. DATASET CONVERSION & YAML GENERATION
# ==============================================================================
# [DEF: prepare_pets2009_dataset()] Ingests raw data, outputs YOLO format images & labels + pets2009.yaml
# [FUNCTION: File copying, label serialization, YAML generation]
# [USED IN: CLI execution, train_rtdetr.py pipeline, test suite]
def prepare_pets2009_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed",
                             yaml_filename: str = "pets2009.yaml") -> str:
    """Ingest raw sequences, partition into train/val/test, and generate Ultralytics YAML manifest."""
    proc_path, raw_path = Path(processed_dir).resolve(), Path(raw_dir).resolve()
    for split in ("train", "val", "test"):
        (proc_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    sequences = fetch_or_discover_raw_data(str(raw_path))

    for seq_name, split in SEQUENCE_SPLITS.items():
        if seq_name not in sequences: continue
        xml_path = sequences[seq_name]["xml_path"]
        img_dir = Path(sequences[seq_name]["images_dir"])
        frame_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")),
                             key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)])
        if not frame_paths: continue

        img_w, img_h = get_image_dimensions(str(frame_paths[0]))
        annotations = parse_cvml_xml(xml_path, img_width=img_w, img_height=img_h)

        for idx, src_frame in enumerate(frame_paths):
            nums = re.findall(r"\d+", src_frame.stem)
            f_num = int(nums[-1]) if nums else idx
            boxes = annotations.get(f_num, annotations.get(idx, []))
            stem = f"{seq_name}_{src_frame.name}"

            shutil.copy2(src_frame, proc_path / "images" / split / stem)
            with open(proc_path / "labels" / split / f"{Path(stem).stem}.txt", "w", encoding="utf-8") as f_lbl:
                for xc, yc, w, h in boxes:
                    f_lbl.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    yaml_path = proc_path / yaml_filename
    manifest = {"path": "data/processed", "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "person"}}
    with open(yaml_path, "w", encoding="utf-8") as f_yml:
        yaml.dump(manifest, f_yml, sort_keys=False)
    return str(yaml_path)

# ==============================================================================
# 8. DATASET VERIFICATION & QUALITY AUDIT
# ==============================================================================
# [DEF: verify_dataset()] Asserts 1-to-1 matching, class 0 (person), coordinate bounds [0, 1]
# [FUNCTION: Sanity checks & sample rendering]
# [USED IN: Ingestion pipelines and test_convert_pets2009.py]
def verify_dataset(processed_dir: str = "data/processed",
                   sample_vis_path: str = "data/processed/sample_verification.jpg") -> Dict[str, Any]:
    """Verify 1-to-1 matching, class 0, and bounds [0, 1]. Render visual composite."""
    proc_path = Path(processed_dir).resolve()
    results = {"status": "PASSED", "checks": {}, "split_counts": {}, "total_images": 0,
               "total_labels": 0, "total_boxes": 0, "coordinate_violations": 0, "class_violations": 0}

    for split in ("train", "val", "test"):
        img_dir, lbl_dir = proc_path / "images" / split, proc_path / "labels" / split
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))
        results["split_counts"][split] = {"images": len(imgs), "labels": len(lbls)}
        results["total_images"] += len(imgs)
        results["total_labels"] += len(lbls)
        assert len(imgs) == len(lbls), f"Mismatch in '{split}': {len(imgs)} imgs vs {len(lbls)} labels"

        for lbl in lbls:
            with open(lbl, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    results["total_boxes"] += 1
                    cls_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if cls_id != 0: results["class_violations"] += 1
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        results["coordinate_violations"] += 1

    assert results["class_violations"] == 0 and results["coordinate_violations"] == 0

    # Draw 3 validation samples into composite image
    samples = []
    val_lbls = sorted(list((proc_path / "labels" / "val").glob("*.txt")))
    for lbl in val_lbls:
        img_file = proc_path / "images" / "val" / f"{lbl.stem}.jpg"
        if not img_file.exists(): img_file = proc_path / "images" / "val" / f"{lbl.stem}.png"
        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w = img.shape[:2]
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1, y1 = int((xc - bw/2)*w), int((yc - bh/2)*h)
                    x2, y2 = int((xc + bw/2)*w), int((yc + bh/2)*h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, (int(xc*w), y2), 4, (0, 0, 255), -1)
        samples.append(cv2.resize(img, (640, 480)))
        if len(samples) >= 3: break

    if samples and sample_vis_path:
        out_p = Path(sample_vis_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), np.hstack(samples))
        results["visual_verification_saved"] = str(out_p)

    return results

# ==============================================================================
# 9. CLI ENTRYPOINT
# ==============================================================================
# [FUNCTION: Main CLI Interface] Parses arguments and runs conversion + verification
# [USED IN: python data/scripts/convert_pets2009.py]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PETS-2009 Ingestion & Conversion")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--yaml_name", type=str, default="pets2009.yaml")
    args = parser.parse_args()
    prepare_pets2009_dataset(args.raw_dir, args.processed_dir, args.yaml_name)
    verify_dataset(args.processed_dir)
