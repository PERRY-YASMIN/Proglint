"""tests/test_convert_pets2009.py - Integration Tests for PETS-2009 Ingestion Pipeline."""

# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import os, sys, tempfile, unittest
# [FUNCTION: Project Path Resolution]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import cv2, numpy as np, yaml
from data.scripts.convert_pets2009 import (
    get_image_dimensions,
    parse_cvml_xml,
    prepare_pets2009_dataset,
    verify_dataset,
)

# ==============================================================================
# 2. INGESTION TEST SUITE
# ==============================================================================
class TestPETS2009Ingestion(unittest.TestCase):
    # [TEST: Header Dimension Extraction] Asserts fast (width, height) without full bitmap load
    def test_01_dynamic_image_dimension_resolution(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            canvas = np.zeros((576, 720, 3), dtype=np.uint8)
            cv2.imwrite(tmp_path, canvas)
            w, h = get_image_dimensions(tmp_path)
            self.assertEqual(w, 720)
            self.assertEqual(h, 576)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # [TEST: XML Parsing & Clamping] Asserts bounding box coordinate normalization into [0.0, 1.0]
    def test_02_cvml_xml_parsing_and_clamping(self):
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
                <box xc="780.0" yc="590.0" w="80.0" h="100.0"/>
            </object>
        </objectlist>
    </frame>
</dataset>"""
            tmp.write(xml_content)

        try:
            boxes = parse_cvml_xml(xml_path, img_width=768, img_height=576)
            self.assertIn(0, boxes)
            frame_0_boxes = boxes[0]
            self.assertEqual(len(frame_0_boxes), 2)

            # Box 1: centered at (384, 288) on 768x576 -> (0.5, 0.5)
            xc1, yc1, w1, h1 = frame_0_boxes[0]
            self.assertAlmostEqual(xc1, 0.5, places=4)
            self.assertAlmostEqual(yc1, 0.5, places=4)

            # Box 2: out-of-bounds box clamped inside [0, 1]
            xc2, yc2, w2, h2 = frame_0_boxes[1]
            self.assertTrue(0.0 <= xc2 <= 1.0)
            self.assertTrue(0.0 <= yc2 <= 1.0)
        finally:
            Path(xml_path).unlink(missing_ok=True)

    # [TEST: Processed Dataset Integrity] Asserts 1-to-1 matching and zero coordinate violations
    def test_03_processed_dataset_integrity(self):
        proc_dir = "data/processed"
        self.assertTrue(Path(proc_dir).exists(), "data/processed does not exist")
        summary = verify_dataset(processed_dir=proc_dir)
        self.assertEqual(summary["status"], "PASSED")
        self.assertEqual(summary["class_violations"], 0)
        self.assertEqual(summary["coordinate_violations"], 0)
        self.assertGreater(summary["total_images"], 0)
        self.assertEqual(summary["total_images"], summary["total_labels"])

    # [TEST: YAML Manifest Schema] Asserts pets2009.yaml has valid keys and class 0: person
    def test_04_yaml_manifest_schema(self):
        yaml_path = Path("data/processed/pets2009.yaml")
        self.assertTrue(yaml_path.exists(), "pets2009.yaml does not exist")
        with open(yaml_path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        self.assertIn("path", manifest)
        self.assertIn("train", manifest)
        self.assertIn("val", manifest)
        self.assertIn("names", manifest)
        self.assertEqual(manifest["names"][0], "person")

    # [TEST: Visual Verification Composite] Asserts sample_verification.jpg exists with 480x1920 shape
    def test_05_visual_verification_artifact(self):
        vis_path = Path("data/processed/sample_verification.jpg")
        self.assertTrue(vis_path.exists(), "sample_verification.jpg was not generated")
        img = cv2.imread(str(vis_path))
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[0], 480)
        self.assertEqual(img.shape[1], 1920)

if __name__ == "__main__":
    unittest.main()
