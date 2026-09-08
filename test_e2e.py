"""
End-to-End Test for the Complete Footfall Tracking System.
Tests FootfallEngine, FastAPI Backend, and Frontend helper pipelines.
"""

import os
import shutil
import tempfile
import time
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app, RUNS_DIR
from frontend.app import (
    extract_first_frame,
    draw_calibration_lines,
    build_occupancy_chart,
)


class TestFullSystemE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_e2e_pipeline(self):
        """Test full pipeline from video creation, calibration, processing, and visualization."""
        temp_dir = tempfile.mkdtemp()
        test_video_path = os.path.join(temp_dir, "sample_clip.mp4")

        try:
            # 1. Generate synthetic video clip (20 frames, 320x240)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(test_video_path, fourcc, 30.0, (320, 240))
            for i in range(20):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                # Draw a white rectangle simulating an object moving downward
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
            form_data = {
                "line_a_norm": "0.40",
                "line_b_norm": "0.60",
                "timeout_sec": "2.5",
            }
            resp = self.client.post("/process_video", files=files, data=form_data)
            self.assertEqual(resp.status_code, 200)

            result = resp.json()
            self.assertEqual(result["status"], "success")
            self.assertIn("video_filename", result)
            self.assertIn("metrics", result)
            self.assertIn("performance", result)
            self.assertIn("events", result)

            processed_filename = result["video_filename"]
            print(f"\n[E2E Video Processed] File: {processed_filename}")
            print(f"[E2E Metrics] {result['metrics']}")
            print(f"[E2E Performance] {result['performance']}")

            # 5. Test Backend API: Stream/Download processed video
            dl_resp = self.client.get(f"/videos/{processed_filename}")
            self.assertEqual(dl_resp.status_code, 200)
            self.assertEqual(dl_resp.headers.get("content-type"), "video/mp4")
            self.assertGreater(len(dl_resp.content), 0)

            # 6. Test Frontend helper: Build occupancy step-chart
            dummy_events = [
                {"frame": 5, "time_sec": 0.2, "id": 1, "type": "IN"},
                {"frame": 12, "time_sec": 0.4, "id": 2, "type": "IN"},
                {"frame": 18, "time_sec": 0.6, "id": 1, "type": "OUT"},
            ]
            fig = build_occupancy_chart(dummy_events, total_duration_sec=1.0)
            self.assertIsNotNone(fig)
            self.assertEqual(len(fig.data), 1)

            # Cleanup video on disk
            vid_disk_path = os.path.join(RUNS_DIR, processed_filename)
            if os.path.exists(vid_disk_path):
                os.remove(vid_disk_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
