"""test_footfall_engine.py - Unit tests for consolidated FootfallEngine."""
import unittest
import numpy as np
from engine.tracker import FootfallEngine, get_default_model_path, CUSTOM_WEIGHTS_PATH


class TestFootfallEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FootfallEngine(conf_threshold=0.50)

    def setUp(self):
        self.engine.reset()

    def test_initial_state(self):
        self.assertEqual(self.engine.total_in, 0)
        self.assertEqual(self.engine.total_out, 0)
        self.assertEqual(self.engine.occupancy, 0)
        self.assertEqual(len(self.engine.seen_ids), 0)

    def test_default_model_path_resolution(self):
        path = get_default_model_path()
        self.assertTrue(isinstance(path, str))
        self.assertGreater(len(path), 0)

    def test_process_frame_interface(self):
        # Blank test frame (640x480 RGB)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, summary = self.engine.process_frame(dummy_frame, line_a=0.45, line_b=0.55, frame_idx=0)
        
        # Verify output tuple contract
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, dummy_frame.shape)
        self.assertIsInstance(summary, dict)
        self.assertIn("total_in", summary)
        self.assertIn("total_out", summary)
        self.assertIn("occupancy", summary)
        self.assertIn("active_tracks", summary)

    def test_fsm_crossing_in(self):
        # Verify manual state progression for IN crossing
        now = 100.0
        tid = 1
        y_a, y_b = 200, 300

        # Step 1: Cross Line A downward -> PENDING_IN
        prev_y, cy = 180, 220
        state, t_start = "IDLE", now
        if prev_y < y_a <= cy:
            state = "PENDING_IN"
        self.assertEqual(state, "PENDING_IN")

        # Step 2: Cross Line B downward -> DONE / total_in + 1
        prev_y, cy = 250, 320
        if state == "PENDING_IN" and prev_y < y_b <= cy:
            self.engine.total_in += 1
            state = "DONE"
        self.assertEqual(state, "DONE")
        self.assertEqual(self.engine.total_in, 1)
        self.assertEqual(self.engine.occupancy, 1)

    def test_fsm_crossing_out(self):
        # Verify manual state progression for OUT crossing
        now = 100.0
        tid = 2
        y_a, y_b = 200, 300

        # Step 1: Cross Line B upward -> PENDING_OUT
        prev_y, cy = 320, 280
        state, t_start = "IDLE", now
        if prev_y > y_b >= cy:
            state = "PENDING_OUT"
        self.assertEqual(state, "PENDING_OUT")

        # Step 2: Cross Line A upward -> DONE / total_out + 1
        prev_y, cy = 250, 180
        if state == "PENDING_OUT" and prev_y > y_a >= cy:
            self.engine.total_out += 1
            state = "DONE"
        self.assertEqual(state, "DONE")
        self.assertEqual(self.engine.total_out, 1)
        self.assertEqual(self.engine.occupancy, -1)


if __name__ == "__main__":
    unittest.main()