"""tests/test_backend_api.py - Unit Test Suite for FastAPI Backend API."""

# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import os, sys, shutil, tempfile, unittest
# [FUNCTION: Project Path Resolution]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2, numpy as np
# [FROM: fastapi.testclient] In-memory HTTP test client
from fastapi.testclient import TestClient
# [FROM: backend.main] Imports the FastAPI app instance and run directory
from backend.main import app, RUNS_DIR

# ==============================================================================
# 2. REST API TEST SUITE
# ==============================================================================
class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # [INIT: Test Client] Binds in-memory HTTP client to FastAPI app
        cls.client = TestClient(app)

    # [TEST: GET /health] Asserts health endpoint returns 200 and expected metadata
    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["detector"], "Domain-Adapted RT-DETR (PETS-2009)")
        self.assertEqual(data["tracker"], "BoT-SORT")

    # [TEST: Bad Extension] Asserts non-video upload returns 400 Bad Request
    def test_invalid_extension(self):
        files = {"video": ("bad.txt", b"dummy content", "text/plain")}
        resp = self.client.post("/process_video", files=files)
        self.assertEqual(resp.status_code, 400)

    # [TEST: Video Processing & Streaming] End-to-end test of POST /process_video and GET /videos
    def test_process_video_and_stream(self):
        tmp = tempfile.mkdtemp()
        vid_path = os.path.join(tmp, "test.mp4")
        try:
            # Generate small 5-frame synthetic video
            w = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
            for _ in range(5):
                w.write(np.zeros((240, 320, 3), dtype=np.uint8))
            w.release()

            # Upload video to endpoint
            with open(vid_path, "rb") as f:
                files = {"video": ("test.mp4", f, "video/mp4")}
                resp = self.client.post("/process_video", files=files)

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
            self.assertIn("metrics", data)

            # Test video download endpoint
            vid_name = data["video_filename"]
            dl_resp = self.client.get(f"/videos/{vid_name}")
            self.assertEqual(dl_resp.status_code, 200)

            # Cleanup video on disk
            disk_file = os.path.join(RUNS_DIR, vid_name)
            if os.path.exists(disk_file):
                os.remove(disk_file)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
