"""
API Test Suite for FastAPI Footfall Service in backend/main.py.
Tests /health, /process_video, /videos/{filename}, error handling, and cleanup.
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app, RUNS_DIR


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check(self):
        """Verify /health endpoint returns CUDA status and device."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("cuda_available", data)
        self.assertIn("device", data)
        print(f"\n[API Test Health] {data}")

    def test_02_invalid_file_extension(self):
        """Verify uploading an invalid file extension triggers HTTP 400."""
        files = {
            "video": ("invalid_file.txt", b"dummy content", "text/plain"),
        }
        data = {
            "line_a_norm": 0.4,
            "line_b_norm": 0.6,
            "timeout_sec": 2.5,
        }
        response = self.client.post("/process_video", files=files, data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file extension", response.json()["detail"])

    def test_03_process_video_and_stream(self):
        """Verify /process_video processes video, encodes H.264, returns schema, and /videos serves file."""
        # Create a small synthetic video
        temp_dir = tempfile.mkdtemp()
        test_video_path = os.path.join(temp_dir, "test_input.mp4")
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(test_video_path, fourcc, 30.0, (320, 240))
            for i in range(10):
                frame = np.full((240, 320, 3), 70 + (i * 10), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            with open(test_video_path, "rb") as f:
                files = {"video": ("test_input.mp4", f, "video/mp4")}
                form_data = {
                    "line_a_norm": "0.45",
                    "line_b_norm": "0.55",
                    "timeout_sec": "2.5",
                }
                response = self.client.post("/process_video", files=files, data=form_data)

            self.assertEqual(response.status_code, 200)
            res_json = response.json()
            self.assertEqual(res_json["status"], "success")
            self.assertIn("video_filename", res_json)
            self.assertIn("metrics", res_json)
            self.assertIn("performance", res_json)
            self.assertIn("events", res_json)

            # Check metrics structure
            metrics = res_json["metrics"]
            self.assertIn("total_in", metrics)
            self.assertIn("total_out", metrics)
            self.assertIn("current_occupancy", metrics)
            self.assertIn("total_tracks", metrics)

            # Check performance structure
            perf = res_json["performance"]
            self.assertIn("avg_fps", perf)
            self.assertEqual(perf["total_frames"], 10)

            video_filename = res_json["video_filename"]
            print(f"[API Process Video Result] video_filename: {video_filename}, avg_fps: {perf['avg_fps']}")

            # Test 4: Download / stream video via GET /videos/{filename}
            video_resp = self.client.get(f"/videos/{video_filename}")
            self.assertEqual(video_resp.status_code, 200)
            self.assertEqual(video_resp.headers.get("content-type"), "video/mp4")
            self.assertGreater(len(video_resp.content), 0)

            # Clean up generated video
            video_file_on_disk = os.path.join(RUNS_DIR, video_filename)
            if os.path.exists(video_file_on_disk):
                os.remove(video_file_on_disk)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_04_get_nonexistent_video(self):
        """Verify GET /videos/{filename} returns 404 for missing file."""
        response = self.client.get("/videos/does_not_exist.mp4")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
