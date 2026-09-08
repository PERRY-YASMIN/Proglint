"""
test_convert_pets2009.py - Unit and Integration Tests for PETS-2009 Ingestion Pipeline.

Tests:
1. CVML XML parsing with dynamic coordinate normalization and boundary clamping.
2. Dynamic image dimension resolution from image headers.
3. Sequence-based dataset partitioning and 1-to-1 image-label alignment.
4. Strict single-class (0) and [0.0, 1.0] bounding box coordinate constraints.
5. YAML manifest schema validation for Ultralytics RT-DETR ingestion.
6. Visual verification artifact generation.
"""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml

from data.scripts.convert_pets2009 import (
    get_image_dimensions,
    parse_cvml_xml,
    prepare_pets2009_dataset,
    verify_dataset,
)


class TestPETS2009Ingestion(unittest.TestCase):
    """Test suite for PETS-2009 dataset conversion and validation."""

    def test_01_dynamic_image_dimension_resolution(self):
        """Verify dynamic header extraction of image dimensions."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a test image with specific non-standard dimensions (720x576)
            canvas = np.zeros((576, 720, 3), dtype=np.uint8)
            cv2.imwrite(tmp_path, canvas)

            w, h = get_image_dimensions(tmp_path)
            self.assertEqual(w, 720)
            self.assertEqual(h, 576)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_02_cvml_xml_parsing_and_clamping(self):
        """Verify CVML XML <box> parsing, coordinate normalization, and boundary clamping."""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False, encoding="utf-8") as tmp:
            xml_path = tmp.name
            xml_content = """<?xml version="1.0" encoding="utf-8"?>
<dataset>
    <frame number="0">
        <objectlist>
            <object id="1">
                <box xc="384.0" yc="288.0" w="60.0" h="120.0"/>
            </object>
            <object id="2">
                <!-- Out of bounds box testing clamping -->
                <box xc="780.0" yc="590.0" w="80.0" h="100.0"/>
            </object>
        </objectlist>
    </frame>
</dataset>"""
            tmp.write(xml_content)

        try:
            # Image size: 768x576
            boxes = parse_cvml_xml(xml_path, img_width=768, img_height=576)
            self.assertIn(0, boxes)
            frame_0_boxes = boxes[0]
            self.assertEqual(len(frame_0_boxes), 2)

            # Box 1: centered at (384, 288) -> normalized (0.5, 0.5)
            xc1, yc1, w1, h1 = frame_0_boxes[0]
            self.assertAlmostEqual(xc1, 0.5, places=4)
            self.assertAlmostEqual(yc1, 0.5, places=4)
            self.assertAlmostEqual(w1, 60.0 / 768.0, places=4)
            self.assertAlmostEqual(h1, 120.0 / 576.0, places=4)

            # Box 2: clamped within [0, 1]
            xc2, yc2, w2, h2 = frame_0_boxes[1]
            self.assertTrue(0.0 <= xc2 <= 1.0)
            self.assertTrue(0.0 <= yc2 <= 1.0)
            self.assertTrue(0.0 <= w2 <= 1.0)
            self.assertTrue(0.0 <= h2 <= 1.0)
        finally:
            Path(xml_path).unlink(missing_ok=True)

    def test_03_processed_dataset_integrity(self):
        """Verify the processed PETS-2009 dataset partitions, alignment, and coordinate constraints."""
        proc_dir = "data/processed"
        self.assertTrue(Path(proc_dir).exists(), "data/processed does not exist")

        summary = verify_dataset(processed_dir=proc_dir)
        self.assertEqual(summary["status"], "PASSED")
        self.assertEqual(summary["class_violations"], 0)
        self.assertEqual(summary["coordinate_violations"], 0)
        self.assertGreater(summary["total_images"], 0)
        self.assertEqual(summary["total_images"], summary["total_labels"])

        # Check split breakdown
        splits = summary["split_counts"]
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)
        self.assertGreater(splits["train"]["images"], 0)
        self.assertGreater(splits["val"]["images"], 0)
        self.assertGreater(splits["test"]["images"], 0)

    def test_04_yaml_manifest_schema(self):
        """Verify pets2009.yaml schema conformity for RT-DETR training."""
        yaml_path = Path("data/processed/pets2009.yaml")
        self.assertTrue(yaml_path.exists(), "pets2009.yaml does not exist")

        with open(yaml_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)

        self.assertIn("path", manifest)
        self.assertIn("train", manifest)
        self.assertIn("val", manifest)
        self.assertIn("test", manifest)
        self.assertIn("names", manifest)
        self.assertEqual(manifest["names"][0], "person")

    def test_05_visual_verification_artifact(self):
        """Verify sample_verification.jpg is created with proper composite resolution."""
        vis_path = Path("data/processed/sample_verification.jpg")
        self.assertTrue(vis_path.exists(), "sample_verification.jpg was not generated")
        self.assertGreater(vis_path.stat().st_size, 10000, "sample_verification.jpg is suspiciously small")

        img = cv2.imread(str(vis_path))
        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 3)
        # 3 horizontal tiles of 640x480 -> 480x1920
        self.assertEqual(img.shape[0], 480)
        self.assertEqual(img.shape[1], 1920)


if __name__ == "__main__":
    unittest.main()
