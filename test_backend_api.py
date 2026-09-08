"""
test_backend_api.py - Streamlined Unit Test Suite for FastAPI Backend API.
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

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["detector"], "Domain-Adapted RT-DETR (PETS-2009)")
        self.assertEqual(data["tracker"], "BoT-SORT")

    def test_invalid_extension(self):
        files = {"video": ("bad.txt", b"dummy content", "text/plain")}
        resp = self.client.post("/process_video", files=files)
        self.assertEqual(resp.status_code, 400)

    def test_process_video_and_stream(self):
        tmp = tempfile.mkdtemp()
        vid_path = os.path.join(tmp, "test.mp4")
        try:
            w = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
            for _ in range(5):
                w.write(np.zeros((240, 320, 3), dtype=np.uint8))
            w.release()

            with open(vid_path, "rb") as f:
                files = {"video": ("test.mp4", f, "video/mp4")}
                resp = self.client.post("/process_video", files=files)

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
            self.assertIn("metrics", data)

            vid_name = data["video_filename"]
            dl_resp = self.client.get(f"/videos/{vid_name}")
            self.assertEqual(dl_resp.status_code, 200)

            disk_file = os.path.join(RUNS_DIR, vid_name)
            if os.path.exists(disk_file):
                os.remove(disk_file)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
