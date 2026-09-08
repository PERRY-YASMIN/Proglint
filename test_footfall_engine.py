"""
test_footfall_engine.py - Streamlined Unit Test Suite for FootfallEngine.
Validates FSM crossing transitions, double-count prevention, frame processing, and video processing.
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from engine.tracker import FootfallEngine, CUSTOM_WEIGHTS_PATH, get_default_model_path
from engine.fsm_counter import DualTripwireFSM, TrackInfo, TrackState


class TestFootfallEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FootfallEngine()

    def setUp(self):
        self.engine.reset()

    def test_initial_state(self):
        self.assertEqual(self.engine.total_in, 0)
        self.assertEqual(self.engine.total_out, 0)
        self.assertEqual(self.engine.occupancy, 0)

    def test_fsm_in_crossing(self):
        """Downward crossing Line A (100) -> Line B (200) triggers IN."""
        fsm = DualTripwireFSM(timeout_sec=4.0)
        t = TrackInfo(track_id=1)
        # Step 1: cross line A (100)
        ev1 = fsm.update_fsm(t, prev_coord=80, curr_coord=120, line_a=100, line_b=200)
        self.assertIsNone(ev1)
        self.assertEqual(t.state, TrackState.PENDING_IN)
        # Step 2: cross line B (200)
        ev2 = fsm.update_fsm(t, prev_coord=180, curr_coord=220, line_a=100, line_b=200)
        self.assertEqual(ev2, "IN")
        self.assertEqual(t.state, TrackState.COMPLETED)

    def test_fsm_out_crossing(self):
        """Upward crossing Line B (200) -> Line A (100) triggers OUT."""
        fsm = DualTripwireFSM(timeout_sec=4.0)
        t = TrackInfo(track_id=2)
        # Step 1: cross line B upwards
        ev1 = fsm.update_fsm(t, prev_coord=220, curr_coord=180, line_a=100, line_b=200)
        self.assertIsNone(ev1)
        self.assertEqual(t.state, TrackState.PENDING_OUT)
        # Step 2: cross line A upwards
        ev2 = fsm.update_fsm(t, prev_coord=120, curr_coord=80, line_a=100, line_b=200)
        self.assertEqual(ev2, "OUT")
        self.assertEqual(t.state, TrackState.COMPLETED)

    def test_double_count_prevention(self):
        """Once COMPLETED, track cannot trigger another count."""
        fsm = DualTripwireFSM(timeout_sec=4.0)
        t = TrackInfo(track_id=3, state=TrackState.COMPLETED)
        ev = fsm.update_fsm(t, prev_coord=180, curr_coord=220, line_a=100, line_b=200)
        self.assertIsNone(ev)

    def test_timeout_reset(self):
        """Pedestrian lingering in gate beyond timeout resets to IDLE."""
        fsm = DualTripwireFSM(timeout_sec=1.0)
        t = TrackInfo(track_id=4)
        fsm.update_fsm(t, prev_coord=80, curr_coord=120, line_a=100, line_b=200, current_time=1.0)
        self.assertEqual(t.state, TrackState.PENDING_IN)
        # 3 seconds later
        fsm.update_fsm(t, prev_coord=120, curr_coord=130, line_a=100, line_b=200, current_time=4.5)
        self.assertEqual(t.state, TrackState.IDLE)

    def test_process_frame(self):
        """Verify process_frame runs on numpy image and returns overlay and summary."""
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        ann, summary = self.engine.process_frame(img, frame_idx=0)
        self.assertEqual(ann.shape, (240, 320, 3))
        self.assertIn("total_in", summary)
        self.assertIn("active_tracks", summary)

    def test_process_video(self):
        """Verify process_video reads and annotates synthetic video."""
        tmp = tempfile.mkdtemp()
        in_path = os.path.join(tmp, "in.mp4")
        out_path = os.path.join(tmp, "out.mp4")
        try:
            w = cv2.VideoWriter(in_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
            for _ in range(5):
                w.write(np.zeros((240, 320, 3), dtype=np.uint8))
            w.release()

            res = self.engine.process_video(in_path, out_path)
            self.assertEqual(res["total_frames"], 5)
            self.assertTrue(os.path.exists(out_path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_weights_error_on_missing(self):
        with self.assertRaises(FileNotFoundError):
            FootfallEngine(model_path="non_existent_model.pt")


if __name__ == "__main__":
    unittest.main()
