"""
Unit and Integration Test Suite for FootfallEngine.
Validates FSM crossing logic, timeouts, hysteresis deadband, garbage collection,
overlays, single-frame processing, and video processing.
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import cv2
import torch

from engine.tracker import (
    FootfallEngine,
    TrackInfo,
    TrackState,
    CUSTOM_WEIGHTS_PATH,
    FALLBACK_WEIGHTS_PATH,
    get_default_model_path,
)


class TestFootfallEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize engine on available device (GPU or CPU)
        cls.engine = FootfallEngine(model_path="yolo11n.pt")
        print(f"\n[Test Setup] Initialized FootfallEngine on device: {cls.engine.device}")

    def setUp(self):
        # Reset state before each test
        self.engine.reset()

    def test_initial_state(self):
        """Verify initial counters and properties."""
        self.assertEqual(self.engine.total_in, 0)
        self.assertEqual(self.engine.total_out, 0)
        self.assertEqual(self.engine.occupancy, 0)
        self.assertEqual(len(self.engine.events), 0)
        self.assertEqual(len(self.engine.tracks), 0)
        self.assertEqual(len(self.engine.seen_track_ids), 0)

    def test_fsm_downward_in_crossing(self):
        """Test pedestrian moving downwards crossing Line A then Line B -> IN count."""
        y_line_a = 400
        y_line_b = 600
        delta_y = 10.0
        timeout_sec = 2.5
        track_id = 101

        # Register track
        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 350.0),
            last_feet_point=(300.0, 350.0),
        )
        self.engine.seen_track_ids.add(track_id)

        # Step 1: Cross Line A downwards (350 -> 420 >= 400 + 10)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=350.0,
            curr_y=420.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=1.0,
            current_frame=2,
        )
        self.assertIsNone(event)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_IN)
        self.assertEqual(self.engine.total_in, 0)

        # Step 2: Cross Line B downwards (420 -> 620 >= 600 + 10)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=420.0,
            curr_y=620.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=1.5,
            current_frame=3,
        )
        self.assertEqual(event, "IN")
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.COUNTED_IN)
        self.assertEqual(self.engine.total_in, 1)
        self.assertEqual(self.engine.occupancy, 1)
        self.assertEqual(len(self.engine.events), 1)
        self.assertEqual(self.engine.events[0]["type"], "IN")
        self.assertEqual(self.engine.events[0]["id"], track_id)

        # Step 3: Verify double-counting lock and transition to COMPLETED
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=620.0,
            curr_y=650.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=2.0,
            current_frame=4,
        )
        self.assertIsNone(event)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.COMPLETED)
        self.assertEqual(self.engine.total_in, 1)

    def test_fsm_upward_out_crossing(self):
        """Test pedestrian moving upwards crossing Line B then Line A -> OUT count."""
        y_line_a = 400
        y_line_b = 600
        delta_y = 10.0
        timeout_sec = 2.5
        track_id = 102

        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 650.0),
            last_feet_point=(300.0, 650.0),
        )
        self.engine.seen_track_ids.add(track_id)

        # Step 1: Cross Line B upwards (650 -> 580 <= 600 - 10)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=650.0,
            curr_y=580.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=1.0,
            current_frame=2,
        )
        self.assertIsNone(event)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_OUT)
        self.assertEqual(self.engine.total_out, 0)

        # Step 2: Cross Line A upwards (580 -> 380 <= 400 - 10)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=580.0,
            curr_y=380.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=1.8,
            current_frame=3,
        )
        self.assertEqual(event, "OUT")
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.COUNTED_OUT)
        self.assertEqual(self.engine.total_out, 1)
        self.assertEqual(self.engine.occupancy, -1)
        self.assertEqual(len(self.engine.events), 1)
        self.assertEqual(self.engine.events[0]["type"], "OUT")

        # Step 3: Verify double-counting lock and transition to COMPLETED
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=380.0,
            curr_y=350.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=2.0,
            current_frame=4,
        )
        self.assertIsNone(event)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.COMPLETED)
        self.assertEqual(self.engine.total_out, 1)

    def test_hysteresis_deadband(self):
        """Jitter before tripwire margin buffer should NOT trigger state transition."""
        y_line_a = 400
        y_line_b = 600
        delta_y = 10.0
        track_id = 103

        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 385.0),
            last_feet_point=(300.0, 385.0),
        )

        # Move from 385 to 392 (jitter remains above tripwire margin 400 - 5 = 395)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=385.0,
            curr_y=392.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=2.5,
            current_time=1.0,
            current_frame=2,
        )
        self.assertIsNone(event)
        # Should remain IDLE because line margin was not reached
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.IDLE)

    def test_velocity_aware_trajectory_crossing_with_margin_buffer(self):
        """Verify velocity-aware trajectory segment math catches discrete jumps across tripwires."""
        y_line_a = 400
        y_line_b = 600
        track_id = 106

        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 395.0),
            last_feet_point=(300.0, 395.0),
        )

        # Discrete step from 395 to 405 (crosses 400 in 1 frame delta)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_y=395.0,
            curr_y=405.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=10.0,
            timeout_sec=4.0,
            current_time=1.0,
            current_frame=2,
        )
        self.assertIsNone(event)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_IN)

        # Discrete step from 595 to 605 across Line B (600)
        event2 = self.engine._update_fsm(
            track_id=track_id,
            prev_y=595.0,
            curr_y=605.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=10.0,
            timeout_sec=4.0,
            current_time=1.5,
            current_frame=3,
        )
        self.assertEqual(event2, "IN")
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.COUNTED_IN)
        self.assertEqual(self.engine.total_in, 1)

    def test_timeout_reset(self):
        """Track in PENDING state should reset to IDLE if timeout_sec exceeded."""
        y_line_a = 400
        y_line_b = 600
        track_id = 104

        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 350.0),
            last_feet_point=(300.0, 350.0),
        )

        # Trigger PENDING_IN at t = 1.0s
        self.engine._update_fsm(
            track_id=track_id,
            prev_y=350.0,
            curr_y=430.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=10.0,
            timeout_sec=2.5,
            current_time=1.0,
            current_frame=2,
        )
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_IN)

        # Loitering between lines at t = 4.0s (3.0s elapsed > 2.5s timeout)
        self.engine._update_fsm(
            track_id=track_id,
            prev_y=430.0,
            curr_y=450.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=10.0,
            timeout_sec=2.5,
            current_time=4.0,
            current_frame=90,
        )
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.IDLE)

    def test_garbage_collection(self):
        """Tracks unseen for > 60 frames should be purged from memory."""
        track_id_active = 201
        track_id_stale = 202

        self.engine.tracks[track_id_active] = TrackInfo(
            track_id=track_id_active,
            last_seen_frame=100,
        )
        self.engine.tracks[track_id_stale] = TrackInfo(
            track_id=track_id_stale,
            last_seen_frame=30,  # 105 - 30 = 75 frames ago (> 60)
        )

        self.engine._purge_stale_tracks(current_frame=105)
        self.assertIn(track_id_active, self.engine.tracks)
        self.assertNotIn(track_id_stale, self.engine.tracks)

    def test_process_frame_interface(self):
        """Verify process_frame runs and returns valid annotated image and dictionary."""
        # Create a dummy 480x640 frame
        dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

        annotated_frame, summary = self.engine.process_frame(
            frame=dummy_frame,
            line_a_norm=0.35,
            line_b_norm=0.65,
            timeout_sec=2.5,
        )

        self.assertEqual(annotated_frame.shape, dummy_frame.shape)
        self.assertIn("frame", summary)
        self.assertIn("total_in", summary)
        self.assertIn("total_out", summary)
        self.assertIn("occupancy", summary)
        self.assertIn("active_tracks", summary)
        self.assertIn("new_events", summary)

    def test_process_video_interface(self):
        """Verify process_video processes video, writes output, and returns expected schema."""
        temp_dir = tempfile.mkdtemp()
        try:
            in_video_path = os.path.join(temp_dir, "input_test.mp4")
            out_video_path = os.path.join(temp_dir, "output_test.mp4")

            # Create a 15-frame synthetic test video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(in_video_path, fourcc, 30.0, (320, 240))
            for i in range(15):
                frame = np.full((240, 320, 3), 50 + (i * 5), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            progress_calls = []

            def progress_cb(current, total, fps):
                progress_calls.append((current, total, fps))

            metrics = self.engine.process_video(
                input_path=in_video_path,
                output_raw_path=out_video_path,
                line_a_norm=0.4,
                line_b_norm=0.6,
                timeout_sec=2.5,
                progress_callback=progress_cb,
            )

            # Assert metrics dictionary schema
            self.assertIn("total_in", metrics)
            self.assertIn("total_out", metrics)
            self.assertIn("occupancy", metrics)
            self.assertIn("total_tracks", metrics)
            self.assertIn("events", metrics)
            self.assertIn("avg_fps", metrics)
            self.assertIn("total_frames", metrics)
            self.assertEqual(metrics["total_frames"], 15)
            self.assertGreater(len(progress_calls), 0)
            self.assertTrue(os.path.exists(out_video_path))
            self.assertGreater(os.path.getsize(out_video_path), 0)
            print(f"[Test Video Metrics] {metrics}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


    def test_retreat_resets_pending_state(self):
        """If a track enters PENDING_IN and retreats back above Line A, state resets to IDLE."""
        y_line_a = 400
        y_line_b = 600
        track_id = 105

        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 350.0),
            last_feet_point=(300.0, 350.0),
        )

        # Crosses Line A downwards -> PENDING_IN
        self.engine._update_fsm(
            track_id=track_id,
            prev_y=350.0,
            curr_y=420.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=10.0,
            timeout_sec=2.5,
            current_time=1.0,
            current_frame=2,
        )
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_IN)

        # Retreats back above Line A (380 <= 400 - 10)
        self.engine._update_fsm(
            track_id=track_id,
            prev_y=420.0,
            curr_y=380.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=10.0,
            timeout_sec=2.5,
            current_time=1.4,
            current_frame=3,
        )
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.IDLE)
        self.assertEqual(self.engine.total_in, 0)

    def test_fast_moving_pedestrian_tripwire_crossing(self):
        """Verify that fast-moving persons jumping across lines in single frames trigger crossings."""
        y_line_a = 400
        y_line_b = 600
        delta_y = 10.0
        timeout_sec = 3.0

        # Case 1: Fast mover jumps across Line A in 1 frame (350 -> 450)
        track_id1 = 301
        self.engine.tracks[track_id1] = TrackInfo(
            track_id=track_id1,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 350.0),
            last_feet_point=(300.0, 350.0),
        )
        event1 = self.engine._update_fsm(
            track_id=track_id1,
            prev_y=350.0,
            curr_y=450.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=1.0,
            current_frame=2,
        )
        self.assertIsNone(event1)
        self.assertEqual(self.engine.tracks[track_id1].state, TrackState.PENDING_IN)

        # Case 2: Fast mover jumps across Line B in 1 frame (450 -> 650) -> COUNTED_IN
        event2 = self.engine._update_fsm(
            track_id=track_id1,
            prev_y=450.0,
            curr_y=650.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=1.2,
            current_frame=3,
        )
        self.assertEqual(event2, "IN")
        self.assertEqual(self.engine.tracks[track_id1].state, TrackState.COUNTED_IN)
        self.assertEqual(self.engine.total_in, 1)

        # Subsequent frame marks as COMPLETED
        self.engine._update_fsm(
            track_id=track_id1,
            prev_y=650.0,
            curr_y=700.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=1.4,
            current_frame=4,
        )
        self.assertEqual(self.engine.tracks[track_id1].state, TrackState.COMPLETED)

        # Case 3: Super fast mover jumps across BOTH lines in 1 single frame (300 -> 650)
        track_id2 = 302
        self.engine.tracks[track_id2] = TrackInfo(
            track_id=track_id2,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(300.0, 300.0),
            last_feet_point=(300.0, 300.0),
        )
        event3 = self.engine._update_fsm(
            track_id=track_id2,
            prev_y=300.0,
            curr_y=650.0,
            y_line_a=y_line_a,
            y_line_b=y_line_b,
            delta_y=delta_y,
            timeout_sec=timeout_sec,
            current_time=2.0,
            current_frame=5,
        )
        self.assertEqual(event3, "IN")
        self.assertEqual(self.engine.tracks[track_id2].state, TrackState.COUNTED_IN)
        self.assertEqual(self.engine.total_in, 2)

    def test_process_video_with_frame_stride(self):
        """Verify process_video with frame_stride=3 executes faster and yields expected schema."""
        temp_dir = tempfile.mkdtemp()
        try:
            in_video_path = os.path.join(temp_dir, "stride_test_in.mp4")
            out_video_path = os.path.join(temp_dir, "stride_test_out.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(in_video_path, fourcc, 30.0, (320, 240))
            for i in range(12):
                frame = np.full((240, 320, 3), 40 + i, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            metrics = self.engine.process_video(
                input_path=in_video_path,
                output_raw_path=out_video_path,
                line_a_norm=0.4,
                line_b_norm=0.6,
                frame_stride=3,
            )

            self.assertEqual(metrics["total_frames"], 12)
            self.assertIn("avg_fps", metrics)
            self.assertTrue(os.path.exists(out_video_path))
            print(f"[Test Frame Stride Metrics] {metrics}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dynamic_checkpoint_resolution_logic(self):
        """Verify get_default_model_path returns custom_train/best.pt if exists, else yolo11n.pt."""
        with patch("os.path.exists") as mock_exists:
            # Case 1: Custom weights present on disk
            mock_exists.side_effect = lambda path: path == CUSTOM_WEIGHTS_PATH
            self.assertEqual(get_default_model_path(), CUSTOM_WEIGHTS_PATH)

            # Case 2: Custom weights absent on disk -> Fallback to yolo11n.pt
            mock_exists.side_effect = lambda path: False
            self.assertEqual(get_default_model_path(), FALLBACK_WEIGHTS_PATH)

    def test_engine_init_defaults_and_explicit_model(self):
        """Verify FootfallEngine initializes with resolved model path or explicit caller path."""
        # Default initialization uses resolved path
        default_engine = FootfallEngine()
        self.assertIn(default_engine.model_path, [CUSTOM_WEIGHTS_PATH, FALLBACK_WEIGHTS_PATH])
        self.assertEqual(default_engine.device, "cuda:0" if torch.cuda.is_available() else "cpu")

        # Explicit model_path is respected
        explicit_engine = FootfallEngine(model_path="yolo11n.pt")
        self.assertEqual(explicit_engine.model_path, "yolo11n.pt")

    def test_detection_caching_and_reuse_on_frame_skipping(self):
        """Verify detection caching on full frames and bbox reuse on skipped frames."""
        dummy_frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        # 1. Full detection frame (skip_detection=False)
        _, summary_full = self.engine.process_frame(
            frame=dummy_frame,
            frame_idx=0,
            skip_detection=False,
        )
        self.assertFalse(summary_full["skipped_detection"])
        self.assertIsInstance(self.engine.cached_detections, list)
        self.assertIsInstance(self.engine.cached_active_boxes, dict)

        # 2. Inject active track
        track_id = 999
        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=0,
            prev_feet_point=(320.0, 200.0),
            last_feet_point=(320.0, 200.0),
            bbox=(300.0, 150.0, 340.0, 250.0),
            velocity=(0.0, 0.0),
        )
        self.engine.cached_active_boxes[track_id] = (300.0, 150.0, 340.0, 250.0)

        # 3. Skipped frame (skip_detection=True) reuses cached box and keeps active track
        _, summary_skip = self.engine.process_frame(
            frame=dummy_frame,
            frame_idx=1,
            skip_detection=True,
        )
        self.assertTrue(summary_skip["skipped_detection"])
        self.assertEqual(summary_skip["active_tracks"], 1)
        self.assertIn(track_id, self.engine.cached_active_boxes)
        self.assertEqual(self.engine.tracks[track_id].bbox, (300.0, 150.0, 340.0, 250.0))

    def test_cv2_circle_integer_casting_and_nan_guards(self):
        """Verify that cv2.circle safely handles numpy floats, skips NaNs, and casts integers."""
        dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

        # Track 1: Numpy floats (e.g. np.float32, np.float64) as feet points
        track_np_floats = TrackInfo(
            track_id=801,
            state=TrackState.IDLE,
            bbox=(np.float32(100.2), np.float32(120.8), np.float32(180.4), np.float32(240.6)),
            last_feet_point=(np.float32(140.3), np.float64(240.6)),
        )

        # Track 2: NaN coordinates in last_feet_point (must not crash, must skip)
        track_nan_feet = TrackInfo(
            track_id=802,
            state=TrackState.PENDING_IN,
            bbox=(200.0, 100.0, 260.0, 220.0),
            last_feet_point=(float("nan"), 220.0),
        )

        # Track 3: NaN coordinates in bbox (must not crash, must skip bbox)
        track_nan_bbox = TrackInfo(
            track_id=803,
            state=TrackState.COUNTED_IN,
            bbox=(float("nan"), 100.0, 260.0, 220.0),
            last_feet_point=(230.0, 220.0),
        )

        tracks = [track_np_floats, track_nan_feet, track_nan_bbox]

        # Should render smoothly without raising any OpenCV overload or ValueError exceptions
        annotated = self.engine._draw_overlays(
            frame=dummy_frame,
            y_line_a=150,
            y_line_b=300,
            current_frame_tracks=tracks,
        )
        self.assertEqual(annotated.shape, dummy_frame.shape)

    def test_process_video_resilience_to_frame_drawing_error(self):
        """Verify process_video completes even if a frame raises an annotation exception."""
        temp_dir = tempfile.mkdtemp()
        try:
            in_path = os.path.join(temp_dir, "robust_test.mp4")
            out_path = os.path.join(temp_dir, "robust_out.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(in_path, fourcc, 30.0, (320, 240))
            for i in range(10):
                writer.write(np.full((240, 320, 3), 50 + i, dtype=np.uint8))
            writer.release()

            original_draw = self.engine._draw_overlays
            call_count = [0]

            def faulty_draw(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 3:
                    raise RuntimeError("Simulated OpenCV rendering crash")
                return original_draw(*args, **kwargs)

            with patch.object(self.engine, "_draw_overlays", side_effect=faulty_draw):
                metrics = self.engine.process_video(
                    input_path=in_path,
                    output_raw_path=out_path,
                    line_a_norm=0.4,
                    line_b_norm=0.6,
                    frame_stride=1,
                )

            # Processing should succeed for all 10 frames despite exception on frame 3
            self.assertEqual(metrics["total_frames"], 10)
            self.assertTrue(os.path.exists(out_path))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_process_video_resolution_downscaling_for_high_res(self):
        """Verify process_video downscales 1080p (>1280px) video to 960 width maintaining aspect ratio."""
        temp_dir = tempfile.mkdtemp()
        try:
            in_path = os.path.join(temp_dir, "highres_test.mp4")
            out_path = os.path.join(temp_dir, "highres_out.mp4")

            # Create a 1920x1080 test video (5 frames)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(in_path, fourcc, 30.0, (1920, 1080))
            for i in range(5):
                writer.write(np.full((1080, 1920, 3), 40 + i, dtype=np.uint8))
            writer.release()

            metrics = self.engine.process_video(
                input_path=in_path,
                output_raw_path=out_path,
                line_a_norm=0.4,
                line_b_norm=0.6,
                frame_stride=1,
            )

            self.assertEqual(metrics["total_frames"], 5)
            self.assertTrue(os.path.exists(out_path))

            # Inspect output video dimensions
            cap_out = cv2.VideoCapture(out_path)
            out_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_out.release()

            # Target width is 960, scale = 960/1920 = 0.5, target height = 1080 * 0.5 = 540
            self.assertEqual(out_w, 960)
            self.assertEqual(out_h, 540)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_fsm_vertical_left_to_right_in_crossing(self):
        """Test pedestrian moving left-to-right crossing vertical Line A then Line B -> IN count."""
        x_line_a = 400
        x_line_b = 600
        delta_x = 10.0
        timeout_sec = 2.5
        track_id = 201

        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(350.0, 500.0),
            last_feet_point=(350.0, 500.0),
        )
        self.engine.seen_track_ids.add(track_id)

        # Step 1: Cross Line A left-to-right (350 -> 420 >= 400 + 5)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=350.0,
            curr_coord=420.0,
            line_a=x_line_a,
            line_b=x_line_b,
            delta=delta_x,
            timeout_sec=timeout_sec,
            current_time=1.0,
            current_frame=2,
            orientation="vertical",
        )
        self.assertIsNone(event)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_IN)
        self.assertEqual(self.engine.total_in, 0)

        # Step 2: Cross Line B left-to-right (420 -> 620 >= 600)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=420.0,
            curr_coord=620.0,
            line_a=x_line_a,
            line_b=x_line_b,
            delta=delta_x,
            timeout_sec=timeout_sec,
            current_time=1.5,
            current_frame=3,
            orientation="vertical",
        )
        self.assertEqual(event, "IN")
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.COUNTED_IN)
        self.assertEqual(self.engine.total_in, 1)
        self.assertEqual(self.engine.occupancy, 1)

    def test_fsm_vertical_right_to_left_out_crossing(self):
        """Test pedestrian moving right-to-left crossing vertical Line B then Line A -> OUT count."""
        x_line_a = 400
        x_line_b = 600
        delta_x = 10.0
        timeout_sec = 2.5
        track_id = 202

        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(650.0, 500.0),
            last_feet_point=(650.0, 500.0),
        )
        self.engine.seen_track_ids.add(track_id)

        # Step 1: Cross Line B right-to-left (650 -> 580 <= 600)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=650.0,
            curr_coord=580.0,
            line_a=x_line_a,
            line_b=x_line_b,
            delta=delta_x,
            timeout_sec=timeout_sec,
            current_time=1.0,
            current_frame=2,
            orientation="vertical",
        )
        self.assertIsNone(event)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_OUT)
        self.assertEqual(self.engine.total_out, 0)

        # Step 2: Cross Line A right-to-left (580 -> 380 <= 400)
        event = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=580.0,
            curr_coord=380.0,
            line_a=x_line_a,
            line_b=x_line_b,
            delta=delta_x,
            timeout_sec=timeout_sec,
            current_time=1.8,
            current_frame=3,
            orientation="vertical",
        )
        self.assertEqual(event, "OUT")
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.COUNTED_OUT)
        self.assertEqual(self.engine.total_out, 1)
        self.assertEqual(self.engine.occupancy, -1)

    def test_vertical_overlay_rendering(self):
        """Verify _draw_overlays draws vertical lines without error."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = self.engine._draw_overlays(
            frame=frame,
            line_a=200,
            line_b=400,
            current_frame_tracks=[],
            orientation="vertical",
        )
        self.assertEqual(annotated.shape, frame.shape)
        # Check cyan color on vertical Line A at x=200, y=240
        self.assertTrue(np.array_equal(annotated[240, 200], self.engine.COLOR_CYAN))
        # Check magenta color on vertical Line B at x=400, y=240
        self.assertTrue(np.array_equal(annotated[240, 400], self.engine.COLOR_MAGENTA))

    def test_both_overlay_rendering(self):
        """Verify _draw_overlays renders both horizontal (Cyan/Magenta) and vertical (Yellow/Orange) gates."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = self.engine._draw_overlays(
            frame=frame,
            line_a_y=100,
            line_b_y=200,
            line_a_x=150,
            line_b_x=300,
            current_frame_tracks=[],
            orientation="both",
        )
        self.assertEqual(annotated.shape, frame.shape)
        # Check Cyan color on horizontal Line A_y at y=100, x=320
        self.assertTrue(np.array_equal(annotated[100, 320], self.engine.COLOR_CYAN))
        # Check Magenta color on horizontal Line B_y at y=200, x=320
        self.assertTrue(np.array_equal(annotated[200, 320], self.engine.COLOR_MAGENTA))
        # Check Yellow color on vertical Line A_x at x=150, y=240
        self.assertTrue(np.array_equal(annotated[240, 150], self.engine.COLOR_YELLOW))
        # Check Orange color on vertical Line B_x at x=300, y=240
        self.assertTrue(np.array_equal(annotated[240, 300], self.engine.COLOR_ORANGE))

    def test_fsm_both_mode_simultaneous_x_and_y_crossings(self):
        """Test pedestrian track undergoing independent crossings on both Y and X axes."""
        track_id = 901
        self.engine.tracks[track_id] = TrackInfo(
            track_id=track_id,
            state=TrackState.IDLE,
            state_y=TrackState.IDLE,
            state_x=TrackState.IDLE,
            last_seen_frame=1,
            prev_feet_point=(150.0, 350.0),
            last_feet_point=(150.0, 350.0),
        )
        self.engine.seen_track_ids.add(track_id)

        # 1. Y-axis Step 1: cross horizontal Line A_y (350 -> 420 >= 400)
        ev_y1 = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=350.0,
            curr_coord=420.0,
            line_a=400,
            line_b=600,
            delta=10.0,
            timeout_sec=4.0,
            current_time=1.0,
            current_frame=2,
            orientation="horizontal",
            axis="y",
        )
        self.assertIsNone(ev_y1)
        self.assertEqual(self.engine.tracks[track_id].state_y, TrackState.PENDING_IN)
        self.assertEqual(self.engine.tracks[track_id].state, TrackState.PENDING_IN)

        # 2. X-axis Step 1: cross vertical Line A_x (150 -> 220 >= 200)
        ev_x1 = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=150.0,
            curr_coord=220.0,
            line_a=200,
            line_b=400,
            delta=10.0,
            timeout_sec=4.0,
            current_time=1.0,
            current_frame=2,
            orientation="vertical",
            axis="x",
        )
        self.assertIsNone(ev_x1)
        self.assertEqual(self.engine.tracks[track_id].state_x, TrackState.PENDING_IN)

        # 3. Y-axis Step 2: cross horizontal Line B_y (420 -> 620 >= 600)
        ev_y2 = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=420.0,
            curr_coord=620.0,
            line_a=400,
            line_b=600,
            delta=10.0,
            timeout_sec=4.0,
            current_time=1.5,
            current_frame=3,
            orientation="horizontal",
            axis="y",
        )
        self.assertEqual(ev_y2, "IN")
        self.assertEqual(self.engine.tracks[track_id].state_y, TrackState.COUNTED_IN)
        self.assertEqual(self.engine.total_in, 1)
        self.assertEqual(self.engine.events[-1]["axis"], "Y")

        # 4. X-axis Step 2: cross vertical Line B_x (220 -> 420 >= 400)
        ev_x2 = self.engine._update_fsm(
            track_id=track_id,
            prev_coord=220.0,
            curr_coord=420.0,
            line_a=200,
            line_b=400,
            delta=10.0,
            timeout_sec=4.0,
            current_time=1.8,
            current_frame=4,
            orientation="vertical",
            axis="x",
        )
        self.assertEqual(ev_x2, "IN")
        self.assertEqual(self.engine.tracks[track_id].state_x, TrackState.COUNTED_IN)
        self.assertEqual(self.engine.total_in, 2)
        self.assertEqual(self.engine.occupancy, 2)
        self.assertEqual(self.engine.events[-1]["axis"], "X")

    def test_process_video_vertical(self):
        """Verify process_video with orientation='vertical' runs end-to-end and returns orientation."""
        temp_dir = tempfile.mkdtemp()
        try:
            in_path = os.path.join(temp_dir, "vert_test.mp4")
            out_path = os.path.join(temp_dir, "vert_out.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(in_path, fourcc, 30.0, (320, 240))
            for i in range(5):
                writer.write(np.full((240, 320, 3), 60 + i, dtype=np.uint8))
            writer.release()

            metrics = self.engine.process_video(
                input_path=in_path,
                output_raw_path=out_path,
                line_a_norm=0.4,
                line_b_norm=0.6,
                frame_stride=1,
                orientation="vertical",
            )
            self.assertEqual(metrics["total_frames"], 5)
            self.assertEqual(metrics["orientation"], "vertical")
            self.assertTrue(os.path.exists(out_path))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_process_video_both_mode(self):
        """Verify process_video with orientation='both' processes dual-axis coordinates end-to-end."""
        temp_dir = tempfile.mkdtemp()
        try:
            in_path = os.path.join(temp_dir, "both_test.mp4")
            out_path = os.path.join(temp_dir, "both_out.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(in_path, fourcc, 30.0, (320, 240))
            for i in range(5):
                writer.write(np.full((240, 320, 3), 70 + i, dtype=np.uint8))
            writer.release()

            metrics = self.engine.process_video(
                input_path=in_path,
                output_raw_path=out_path,
                line_a_norm=0.45,
                line_b_norm=0.55,
                line_a_y_norm=0.40,
                line_b_y_norm=0.60,
                line_a_x_norm=0.35,
                line_b_x_norm=0.65,
                frame_stride=1,
                orientation="both",
            )
            self.assertEqual(metrics["total_frames"], 5)
            self.assertEqual(metrics["orientation"], "both")
            self.assertTrue(os.path.exists(out_path))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
