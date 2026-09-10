"""tests/test_e2e.py - End-to-End System Test (Engine + Backend + Frontend Helpers)."""

# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import os, sys, shutil, tempfile, unittest
# [FUNCTION: Project Path Resolution]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2, numpy as np
from fastapi.testclient import TestClient
from backend.main import app, RUNS_DIR
from frontend.app import extract_first_frame, draw_calibration_lines, build_occupancy_chart

# ==============================================================================
# 2. END-TO-END PIPELINE TEST
# ==============================================================================
class TestFullSystemE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_e2e_pipeline(self):
        """Full pipeline test: video synthesis -> calibration preview -> API processing -> chart rendering."""
        temp_dir = tempfile.mkdtemp()
        test_video_path = os.path.join(temp_dir, "sample_clip.mp4")

        try:
            # 1. Generate 20-frame synthetic test video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(test_video_path, fourcc, 30.0, (320, 240))
            for i in range(20):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.rectangle(frame, (130, 40 + i * 8), (170, 90 + i * 8), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

            with open(test_video_path, "rb") as f:
                video_bytes = f.read()

            # 2. Test Frontend helper: Extract first frame
            first_frame = extract_first_frame(video_bytes, ".mp4")
            self.assertIsNotNone(first_frame)
            self.assertEqual(first_frame.shape, (240, 320, 3))

            # 3. Test Frontend helper: Draw calibration lines
            preview_rgb = draw_calibration_lines(first_frame, line_a_norm=0.4, line_b_norm=0.6)
            self.assertEqual(preview_rgb.shape, (240, 320, 3))

            # 4. Test Backend API: Process Video
            files = {"video": ("sample_clip.mp4", video_bytes, "video/mp4")}
            form_data = {"line_a_norm": "0.40", "line_b_norm": "0.60", "timeout_sec": "2.5"}
            resp = self.client.post("/process_video", files=files, data=form_data)
            self.assertEqual(resp.status_code, 200)

            result = resp.json()
            self.assertEqual(result["status"], "success")
            self.assertIn("video_filename", result)
            self.assertIn("metrics", result)

            processed_filename = result["video_filename"]

            # 5. Test Backend API: Video stream endpoint
            dl_resp = self.client.get(f"/videos/{processed_filename}")
            self.assertEqual(dl_resp.status_code, 200)

            # 6. Test Frontend helper: Build occupancy chart
            dummy_events = [
                {"frame": 5, "time_sec": 0.2, "id": 1, "type": "IN"},
                {"frame": 18, "time_sec": 0.6, "id": 1, "type": "OUT"}
            ]
            fig = build_occupancy_chart(dummy_events, total_duration_sec=1.0)
            self.assertIsNotNone(fig)

            # Clean up disk video
            vid_disk_path = os.path.join(RUNS_DIR, processed_filename)
            if os.path.exists(vid_disk_path):
                os.remove(vid_disk_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
