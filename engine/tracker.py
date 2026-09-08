"""
FootfallEngine - Real-Time Person Counting and Tracking Engine.

Production-grade tracking and directional footfall analysis using
Ultralytics YOLO11 and ByteTrack with Dual Virtual Tripwires,
Finite State Machine (FSM) crossing logic, Hysteresis Deadband,
and Memory Garbage Collection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configure module-level logger
logger = logging.getLogger("FootfallEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TrackState(str, Enum):
    """Finite State Machine states for pedestrian directional tripwire crossing."""

    IDLE = "IDLE"
    PENDING_IN = "PENDING_IN"
    PENDING_OUT = "PENDING_OUT"
    COUNTED_IN = "COUNTED_IN"
    COUNTED_OUT = "COUNTED_OUT"
    COMPLETED = "COMPLETED"


@dataclass
class TrackInfo:
    """Internal state and trajectory metadata for an active track ID."""

    track_id: int
    state: TrackState = TrackState.IDLE
    state_start_time: float = 0.0
    state_start_frame: int = 0
    last_seen_frame: int = 0
    prev_feet_point: Optional[Tuple[float, float]] = None
    last_feet_point: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels/frame
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))


CUSTOM_WEIGHTS_PATH: str = "runs/custom_train/best.pt"
FALLBACK_WEIGHTS_PATH: str = "yolo11n.pt"


def get_default_model_path() -> str:
    """Resolve default YOLO model weights: custom fine-tuned weights if present, else fallback."""
    if os.path.exists(CUSTOM_WEIGHTS_PATH):
        return CUSTOM_WEIGHTS_PATH
    return FALLBACK_WEIGHTS_PATH


class FootfallEngine:
    """Production-grade person counting and tracking engine.

    Leverages YOLO11 pedestrian detection (class 0) and ByteTrack multi-object
    tracking, paired with dual virtual tripwires and a robust Finite State Machine (FSM)
    with hysteresis deadband to achieve high-accuracy real-time bidirectional footfall counting.
    Features dynamic custom fine-tuned weights loading and high-throughput frame-skipping.
    """

    # BGR Color Constants
    COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)       # Line A (Outer/Entrance)
    COLOR_MAGENTA: Tuple[int, int, int] = (255, 0, 255)   # Line B (Inner/Exit)
    COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)       # Counted / Completed
    COLOR_AMBER: Tuple[int, int, int] = (0, 191, 255)     # Pending state
    COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)   # Default labels / text
    COLOR_DARK_BG: Tuple[int, int, int] = (20, 20, 20)    # Dashboard background

    def __init__(
        self,
        model_path: Optional[str] = (
            CUSTOM_WEIGHTS_PATH if os.path.exists(CUSTOM_WEIGHTS_PATH) else FALLBACK_WEIGHTS_PATH
        ),
        device: Optional[str] = None,
        conf_threshold: float = 0.30,
        tracker_config: str = "bytetrack.yaml",
        purge_timeout_frames: int = 60,
    ) -> None:
        """Initialize the FootfallEngine with YOLO11 model and ByteTrack configuration.

        Args:
            model_path: Path or identifier for Ultralytics YOLO model. If None, dynamically
                        checks if 'runs/custom_train/best.pt' exists on disk; if so, defaults
                        to it. Otherwise, falls back to 'yolo11n.pt'.
            device: Compute device ('cuda:0', 'cpu', etc.). If None, automatically defaults
                    to CUDA if torch.cuda.is_available() else CPU.
            conf_threshold: Minimum detection confidence threshold (defaults to 0.30 for occlusion/blur preservation).
            tracker_config: Tracking configuration file (defaults to 'bytetrack.yaml').
            purge_timeout_frames: Maximum frames a track can remain unseen before being purged.
        """
        # Dynamic checkpoint resolution
        if model_path is None:
            resolved_model_path = get_default_model_path()
        elif not os.path.exists(model_path) and model_path == CUSTOM_WEIGHTS_PATH:
            logger.warning(
                "Custom weights '%s' not found on disk. Falling back to base model '%s'.",
                CUSTOM_WEIGHTS_PATH,
                FALLBACK_WEIGHTS_PATH,
            )
            resolved_model_path = FALLBACK_WEIGHTS_PATH
        else:
            resolved_model_path = model_path

        self.model_path: str = resolved_model_path

        if device is None:
            self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Initializing FootfallEngine with model='%s' on device='%s'...",
            self.model_path,
            self.device,
        )

        self.model: YOLO = YOLO(self.model_path)
        self.conf_threshold: float = float(conf_threshold)
        self.tracker_config: str = tracker_config
        self.purge_timeout_frames: int = int(purge_timeout_frames)

        # Counting and State Metrics
        self.total_in: int = 0
        self.total_out: int = 0
        self.events: List[Dict[str, Any]] = []

        # Active tracking history keyed by track_id
        self.tracks: Dict[int, TrackInfo] = {}
        # Set of all unique track IDs ever registered in this session
        self.seen_track_ids: Set[int] = set()

        # Active confirmed track IDs from ByteTrack & last detection frame
        self.active_track_ids: Set[int] = set()
        self.last_detection_frame: Optional[int] = None

        # Cached detections and active track bounding boxes for high-throughput frame skipping
        self.cached_detections: List[Dict[str, Any]] = []
        self.cached_active_boxes: Dict[int, Tuple[float, float, float, float]] = {}

        # Internal frame counter
        self.current_frame: int = 0

        logger.info("FootfallEngine successfully initialized and ready.")

    @property
    def occupancy(self) -> int:
        """Current net occupancy (total_in - total_out)."""
        return self.total_in - self.total_out

    def reset(self) -> None:
        """Reset all counters, active tracks, history, and logged events."""
        self.total_in = 0
        self.total_out = 0
        self.events.clear()
        self.tracks.clear()
        self.seen_track_ids.clear()
        self.active_track_ids.clear()
        self.last_detection_frame = None
        self.cached_detections.clear()
        self.cached_active_boxes.clear()
        self.current_frame = 0
        logger.info("FootfallEngine states and counters reset.")

    def get_counts(self) -> Dict[str, int]:
        """Return a snapshot of current counting metrics."""
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "occupancy": self.occupancy,
            "active_tracks": len(self.tracks),
            "total_unique_tracks": len(self.seen_track_ids),
        }

    def get_events(self) -> List[Dict[str, Any]]:
        """Return a copy of the chronological crossing events list."""
        return list(self.events)

    def _purge_stale_tracks(self, current_frame: int) -> None:
        """Purge track IDs not seen for more than purge_timeout_frames to prevent memory leaks."""
        stale_ids = [
            t_id
            for t_id, info in self.tracks.items()
            if (current_frame - info.last_seen_frame) > self.purge_timeout_frames
        ]
        for t_id in stale_ids:
            del self.tracks[t_id]
        if stale_ids:
            logger.debug(
                "Frame %d: Purged %d stale tracks: %s",
                current_frame,
                len(stale_ids),
                stale_ids,
            )

    def _update_fsm(
        self,
        track_id: int,
        prev_y: Optional[float] = None,
        curr_y: Optional[float] = None,
        y_line_a: Optional[int] = None,
        y_line_b: Optional[int] = None,
        delta_y: float = 10.0,
        timeout_sec: float = 4.0,
        current_time: float = 0.0,
        current_frame: int = 0,
        orientation: str = "horizontal",
        *,
        prev_coord: Optional[float] = None,
        curr_coord: Optional[float] = None,
        line_a: Optional[int] = None,
        line_b: Optional[int] = None,
        delta: Optional[float] = None,
    ) -> Optional[str]:
        """Update Finite State Machine for a given track ID and return crossing type if triggered.

        Supports both horizontal (Y-axis motion) and vertical (X-axis motion) tripwire gating.
        Uses velocity-aware trajectory segment crossing with a 5-pixel margin buffer:
        - Forward crossing (Down / Right): (prev < line <= curr) or (curr > prev and prev <= line + 5 and curr >= line - 5)
        - Backward crossing (Up / Left): (prev > line >= curr) or (curr < prev and prev >= line - 5 and curr <= line + 5)
        - Double-count prevention permanently locks COMPLETED tracks until purged.

        Args:
            track_id: Tracking ID of pedestrian.
            prev_y: Previous feet Y/motion coordinate (or None if first appearance).
            curr_y: Current feet Y/motion coordinate.
            y_line_a: Pixel coordinate of Line A.
            y_line_b: Pixel coordinate of Line B.
            delta_y: Hysteresis deadband in pixels.
            timeout_sec: Timeout in seconds for pending crossing states (default 4.0s).
            current_time: Current timestamp in seconds.
            current_frame: Current frame index.
            orientation: "horizontal" or "vertical".
            prev_coord: Generalized previous coordinate along motion axis.
            curr_coord: Generalized current coordinate along motion axis.
            line_a: Generalized pixel coordinate of Line A.
            line_b: Generalized pixel coordinate of Line B.
            delta: Generalized hysteresis deadband in pixels.

        Returns:
            "IN" if an IN crossing was finalized, "OUT" if OUT was finalized, else None.
        """
        p_coord = prev_coord if prev_coord is not None else prev_y
        c_coord = curr_coord if curr_coord is not None else curr_y
        l_a = line_a if line_a is not None else y_line_a
        l_b = line_b if line_b is not None else y_line_b
        d_val = delta if delta is not None else delta_y

        track = self.tracks[track_id]

        # Double-Count Prevention: Once COMPLETED, the track ID is permanently locked
        if track.state == TrackState.COMPLETED:
            return None

        # If previously COUNTED in the preceding frame, transition to COMPLETED to maintain FSM sequence
        if track.state in (TrackState.COUNTED_IN, TrackState.COUNTED_OUT):
            track.state = TrackState.COMPLETED
            return None

        # Check for timeout on pending states (default 4.0s)
        if track.state in (TrackState.PENDING_IN, TrackState.PENDING_OUT):
            if (current_time - track.state_start_time) > timeout_sec:
                logger.debug(
                    "Track %d: Pending state %s timed out (> %.2fs). Resetting to IDLE.",
                    track_id,
                    track.state.value,
                    timeout_sec,
                )
                track.state = TrackState.IDLE

        if c_coord is None or l_a is None or l_b is None:
            return None

        # First detection: no trajectory movement exists yet
        if p_coord is None:
            return None

        # Determine spatial orientation between Line A (outer) and Line B (inner)
        line_a_is_first = l_a <= l_b
        first_line = min(l_a, l_b)
        second_line = max(l_a, l_b)

        crossing_event: Optional[str] = None

        # Velocity-Aware Trajectory Segment Crossing Check with 5-pixel margin buffer
        def _crosses_forward(line_val: float, margin: float = 5.0) -> bool:
            return (p_coord < line_val <= c_coord) or (
                c_coord > p_coord and p_coord <= (line_val + margin) and c_coord >= (line_val - margin)
            )

        def _crosses_backward(line_val: float, margin: float = 5.0) -> bool:
            return (p_coord > line_val >= c_coord) or (
                c_coord < p_coord and p_coord >= (line_val - margin) and c_coord <= (line_val + margin)
            )

        if line_a_is_first:
            # Line A is first_line (Top or Left), Line B is second_line (Bottom or Right)
            # Forward motion (increasing coord: Top->Bottom or Left->Right) = IN (Line A -> Line B)
            # Backward motion (decreasing coord: Bottom->Top or Right->Left) = OUT (Line B -> Line A)

            if track.state == TrackState.IDLE:
                # Fast Mover Direct Jump: Crossed both lines in 1 frame
                if (p_coord < first_line and c_coord >= second_line) or (_crosses_forward(first_line) and c_coord >= second_line):
                    track.state = TrackState.COUNTED_IN
                    crossing_event = "IN"
                elif (p_coord > second_line and c_coord <= first_line) or (_crosses_backward(second_line) and c_coord <= first_line):
                    track.state = TrackState.COUNTED_OUT
                    crossing_event = "OUT"
                # Step 1 IN: Crossed Line A forward
                elif _crosses_forward(first_line):
                    track.state = TrackState.PENDING_IN
                    track.state_start_time = current_time
                    track.state_start_frame = current_frame
                    logger.debug("Track %d -> PENDING_IN at frame %d", track_id, current_frame)
                # Step 1 OUT: Crossed Line B backward
                elif _crosses_backward(second_line):
                    track.state = TrackState.PENDING_OUT
                    track.state_start_time = current_time
                    track.state_start_frame = current_frame
                    logger.debug("Track %d -> PENDING_OUT at frame %d", track_id, current_frame)

            elif track.state == TrackState.PENDING_IN:
                # Step 2 IN: Subsequently crossed Line B forward within timeout_sec
                if _crosses_forward(second_line) or (c_coord >= second_line):
                    track.state = TrackState.COUNTED_IN
                    crossing_event = "IN"
                # Retreat / turnaround: Moved back behind Line A
                elif (c_coord <= (first_line - d_val)) or (p_coord >= first_line > c_coord):
                    track.state = TrackState.IDLE
                    logger.debug("Track %d retreated behind Line A. State reset to IDLE.", track_id)

            elif track.state == TrackState.PENDING_OUT:
                # Step 2 OUT: Subsequently crossed Line A backward within timeout_sec
                if _crosses_backward(first_line) or (c_coord <= first_line):
                    track.state = TrackState.COUNTED_OUT
                    crossing_event = "OUT"
                # Retreat / turnaround: Moved back past Line B
                elif (c_coord >= (second_line + d_val)) or (p_coord <= second_line < c_coord):
                    track.state = TrackState.IDLE
                    logger.debug("Track %d retreated past Line B. State reset to IDLE.", track_id)

        else:
            # Inverted orientation: Line A is second_line (Bottom or Right), Line B is first_line (Top or Left)
            # Backward motion (decreasing coord: Bottom->Top or Right->Left) = IN (Line A -> Line B)
            # Forward motion (increasing coord: Top->Bottom or Left->Right) = OUT (Line B -> Line A)

            if track.state == TrackState.IDLE:
                if (p_coord > second_line and c_coord <= first_line) or (_crosses_backward(second_line) and c_coord <= first_line):
                    track.state = TrackState.COUNTED_IN
                    crossing_event = "IN"
                elif (p_coord < first_line and c_coord >= second_line) or (_crosses_forward(first_line) and c_coord >= second_line):
                    track.state = TrackState.COUNTED_OUT
                    crossing_event = "OUT"
                elif _crosses_backward(second_line):
                    track.state = TrackState.PENDING_IN
                    track.state_start_time = current_time
                    track.state_start_frame = current_frame
                    logger.debug("Track %d -> PENDING_IN (inverted) at frame %d", track_id, current_frame)
                elif _crosses_forward(first_line):
                    track.state = TrackState.PENDING_OUT
                    track.state_start_time = current_time
                    track.state_start_frame = current_frame
                    logger.debug("Track %d -> PENDING_OUT (inverted) at frame %d", track_id, current_frame)

            elif track.state == TrackState.PENDING_IN:
                if _crosses_backward(first_line) or (c_coord <= first_line):
                    track.state = TrackState.COUNTED_IN
                    crossing_event = "IN"
                elif (c_coord >= (second_line + d_val)) or (p_coord <= second_line < c_coord):
                    track.state = TrackState.IDLE

            elif track.state == TrackState.PENDING_OUT:
                if _crosses_forward(second_line) or (c_coord >= second_line):
                    track.state = TrackState.COUNTED_OUT
                    crossing_event = "OUT"
                elif (c_coord <= (first_line - d_val)) or (p_coord >= first_line > c_coord):
                    track.state = TrackState.IDLE

        if crossing_event == "IN":
            self.total_in += 1
            event_record = {
                "frame": int(current_frame),
                "time_sec": float(round(current_time, 3)),
                "id": int(track_id),
                "type": "IN",
            }
            self.events.append(event_record)
            logger.info("Crossing Event Registered: %s (Track ID: %d)", event_record, track_id)

        elif crossing_event == "OUT":
            self.total_out += 1
            event_record = {
                "frame": int(current_frame),
                "time_sec": float(round(current_time, 3)),
                "id": int(track_id),
                "type": "OUT",
            }
            self.events.append(event_record)
            logger.info("Crossing Event Registered: %s (Track ID: %d)", event_record, track_id)

        return crossing_event

    def _draw_overlays(
        self,
        frame: np.ndarray,
        line_a: Optional[int] = None,
        line_b: Optional[int] = None,
        current_frame_tracks: Optional[List[TrackInfo]] = None,
        orientation: str = "horizontal",
        *,
        y_line_a: Optional[int] = None,
        y_line_b: Optional[int] = None,
    ) -> np.ndarray:
        """Render tripwires, bounding boxes, feet markers, and top-left HUD dashboard.

        Args:
            frame: Raw BGR input frame.
            line_a: Pixel coordinate for Line A (Y if horizontal, X if vertical).
            line_b: Pixel coordinate for Line B (Y if horizontal, X if vertical).
            current_frame_tracks: List of TrackInfo instances detected in the current frame.
            orientation: "horizontal" or "vertical".
            y_line_a: Backward compatibility alias for line_a.
            y_line_b: Backward compatibility alias for line_b.

        Returns:
            Annotated frame image.
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        tracks = current_frame_tracks if current_frame_tracks is not None else []

        l_a = line_a if line_a is not None else (y_line_a if y_line_a is not None else int(height * 0.45))
        l_b = line_b if line_b is not None else (y_line_b if y_line_b is not None else int(height * 0.55))

        clean_orientation = orientation.lower().strip() if orientation else "horizontal"
        if clean_orientation == "vertical":
            # 1. Draw Dual Virtual Tripwires vertically across full frame height
            # Line A - Cyan (BGR: 255, 255, 0)
            x_a = int(max(0, min(width - 1, round(float(l_a)))))
            cv2.line(annotated, (x_a, 0), (x_a, int(height - 1)), self.COLOR_CYAN, 2, cv2.LINE_AA)
            label_x_a = int(max(10, min(width - 150, x_a + 8)))
            cv2.putText(
                annotated,
                "Line A (Outer)",
                (label_x_a, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.COLOR_CYAN,
                2,
                cv2.LINE_AA,
            )

            # Line B - Magenta (BGR: 255, 0, 255)
            x_b = int(max(0, min(width - 1, round(float(l_b)))))
            cv2.line(annotated, (x_b, 0), (x_b, int(height - 1)), self.COLOR_MAGENTA, 2, cv2.LINE_AA)
            label_x_b = int(max(10, min(width - 150, x_b + 8)))
            cv2.putText(
                annotated,
                "Line B (Inner)",
                (label_x_b, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.COLOR_MAGENTA,
                2,
                cv2.LINE_AA,
            )
        else:
            # 1. Draw Dual Virtual Tripwires horizontally across full frame width
            # Line A - Cyan (BGR: 255, 255, 0)
            y_a = int(max(0, min(height - 1, round(float(l_a)))))
            cv2.line(annotated, (0, y_a), (int(width - 1), y_a), self.COLOR_CYAN, 2, cv2.LINE_AA)
            label_y_a = int(max(20, min(height - 5, y_a - 8)))
            cv2.putText(
                annotated,
                "Line A (Outer)",
                (15, label_y_a),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.COLOR_CYAN,
                2,
                cv2.LINE_AA,
            )

            # Line B - Magenta (BGR: 255, 0, 255)
            y_b = int(max(0, min(height - 1, round(float(l_b)))))
            cv2.line(annotated, (0, y_b), (int(width - 1), y_b), self.COLOR_MAGENTA, 2, cv2.LINE_AA)
            label_y_b = int(max(20, min(height - 5, y_b - 8)))
            cv2.putText(
                annotated,
                "Line B (Inner)",
                (15, label_y_b),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.COLOR_MAGENTA,
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Person Bounding Boxes, ID tags, and Feet Points
        for track in current_frame_tracks:
            if track.bbox is None:
                continue

            try:
                raw_x1, raw_y1, raw_x2, raw_y2 = track.bbox
                if np.isnan(raw_x1) or np.isnan(raw_y1) or np.isnan(raw_x2) or np.isnan(raw_y2):
                    continue

                # Strictly cast to Python int and wrap in bounds check: 0 <= x < width, 0 <= y < height
                x1 = int(max(0, min(width - 1, round(float(raw_x1)))))
                y1 = int(max(0, min(height - 1, round(float(raw_y1)))))
                x2 = int(max(0, min(width - 1, round(float(raw_x2)))))
                y2 = int(max(0, min(height - 1, round(float(raw_y2)))))
                if x2 <= x1 or y2 <= y1:
                    continue
            except (ValueError, TypeError):
                continue

            # Choose bounding box color based on FSM state
            if track.state in (TrackState.COMPLETED, TrackState.COUNTED_IN, TrackState.COUNTED_OUT):
                box_color = self.COLOR_GREEN
            elif track.state in (TrackState.PENDING_IN, TrackState.PENDING_OUT):
                box_color = self.COLOR_AMBER
            else:
                box_color = (240, 200, 70)  # Light cyan-amber for IDLE

            # Thin bounding box strictly cast to integer tuples (thickness=2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # ID label with solid backdrop badge
            label = f"ID: {track.track_id}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_w = int(text_w)
            text_h = int(text_h)
            badge_y1 = int(max(0, min(height - 1, y1 - text_h - 6)))
            badge_y2 = int(max(0, min(height - 1, badge_y1 + text_h + 6)))
            badge_x1 = int(max(0, min(width - 1, x1)))
            badge_x2 = int(max(0, min(width - 1, badge_x1 + text_w + 6)))

            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                box_color,
                -1,
            )
            text_x = int(max(0, min(width - 1, badge_x1 + 3)))
            text_y = int(max(0, min(height - 1, badge_y1 + text_h + 2)))
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Feet coordinate marker (bottom-center):
            # cx = int(round((x1 + x2) / 2.0))
            # cy = int(round(y2))
            try:
                cx = int(round((x1 + x2) / 2.0))
                cy = int(round(y2))
                if 0 <= cx < width and 0 <= cy < height:
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)
            except (ValueError, TypeError):
                pass

        # 3. Persistent Top-Left High-Contrast Dashboard Overlay
        # "IN: <count> | OUT: <count> | OCCUPANCY: <in - out>"
        dashboard_text = (
            f"IN: {self.total_in} | OUT: {self.total_out} | OCCUPANCY: {self.occupancy}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        font_thick = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            dashboard_text, font, font_scale, font_thick
        )
        text_w = int(text_w)
        text_h = int(text_h)

        pad_x, pad_y = 12, 10
        box_x1 = int(max(0, min(width - 1, 15)))
        box_y1 = int(max(0, min(height - 1, 15)))
        box_x2 = int(max(0, min(width - 1, box_x1 + text_w + (pad_x * 2))))
        box_y2 = int(max(0, min(height - 1, box_y1 + text_h + (pad_y * 2))))

        # Alpha-blended dark backdrop for high contrast
        if box_x2 > box_x1 and box_y2 > box_y1:
            sub_img = annotated[box_y1:box_y2, box_x1:box_x2]
            if sub_img.size > 0:
                dark_rect = np.full_like(sub_img, self.COLOR_DARK_BG)
                alpha = 0.75
                cv2.addWeighted(dark_rect, alpha, sub_img, 1.0 - alpha, 0, sub_img)
                annotated[box_y1:box_y2, box_x1:box_x2] = sub_img

                # Subtle border around dashboard card
                cv2.rectangle(
                    annotated,
                    (box_x1, box_y1),
                    (box_x2, box_y2),
                    (80, 80, 80),
                    1,
                    cv2.LINE_AA,
                )

        # Text inside dashboard
        text_pos_x = int(max(0, min(width - 1, box_x1 + pad_x)))
        text_pos_y = int(max(0, min(height - 1, box_y1 + text_h + pad_y - 2)))
        cv2.putText(
            annotated,
            dashboard_text,
            (text_pos_x, text_pos_y),
            font,
            font_scale,
            self.COLOR_WHITE,
            font_thick,
            cv2.LINE_AA,
        )

        return annotated

    def process_frame(
        self,
        frame: np.ndarray,
        line_a_norm: float = 0.45,
        line_b_norm: float = 0.55,
        timeout_sec: float = 4.0,
        delta_y: float = 10.0,
        frame_idx: Optional[int] = None,
        timestamp: Optional[float] = None,
        skip_detection: bool = False,
        orientation: str = "horizontal",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single video frame for person detection, tracking, and counting.

        Args:
            frame: Input image array (BGR format).
            line_a_norm: Normalized coordinate for Line A (0.0 to 1.0).
            line_b_norm: Normalized coordinate for Line B (0.0 to 1.0).
            timeout_sec: Timeout in seconds before resetting pending crossing state (default 4.0s).
            delta_y: Hysteresis deadband in pixels.
            frame_idx: Optional frame index (auto-increments if None).
            timestamp: Optional timestamp in seconds (uses current time / FPS if None).
            skip_detection: When True, skips YOLO11 forward pass and reuses verified tracks without velocity drift.
            orientation: "horizontal" (Top/Bottom flow) or "vertical" (Left/Right flow).

        Returns:
            Tuple of (annotated_frame, frame_summary_dict).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame supplied to process_frame (None or empty array).")

        # Advance frame counters and temporal reference
        if frame_idx is not None:
            self.current_frame = int(frame_idx)
        else:
            self.current_frame += 1

        current_frame = self.current_frame
        current_time = float(timestamp) if timestamp is not None else time.time()

        clean_orientation = orientation.lower().strip() if orientation else "horizontal"
        if clean_orientation != "vertical":
            clean_orientation = "horizontal"

        height, width = frame.shape[:2]
        if clean_orientation == "vertical":
            line_a_pixel = int(round(float(line_a_norm * width)))
            line_b_pixel = int(round(float(line_b_norm * width)))
        else:
            line_a_pixel = int(round(float(line_a_norm * height)))
            line_b_pixel = int(round(float(line_b_norm * height)))

        current_frame_tracks: List[TrackInfo] = []
        new_events: List[Dict[str, Any]] = []

        if not skip_detection:
            # Clear previous frame detection cache
            self.cached_detections.clear()
            self.cached_active_boxes.clear()

            # 1. Full detection: Run YOLO11 pedestrian tracking strictly on class 0 (person)
            # High-resolution inference (imgsz=960 or 1080), conf=0.30, iou=0.45 for small pedestrians and occlusions
            det_conf = float(self.conf_threshold)
            target_imgsz = 1080 if max(height, width) >= 1080 else 960
            results = self.model.track(
                source=frame,
                persist=True,
                classes=[0],
                tracker=self.tracker_config,
                conf=det_conf,
                iou=0.45,
                imgsz=target_imgsz,
                device=self.device,
                verbose=False,
            )

            # 2. Extract bounding boxes and track IDs if any detected
            current_detected_ids: Set[int] = set()
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    xyxy_coords = boxes.xyxy.cpu().numpy()
                    confs = (
                        boxes.conf.cpu().numpy()
                        if boxes.conf is not None
                        else [1.0] * len(track_ids)
                    )

                    for i, track_id in enumerate(track_ids):
                        current_detected_ids.add(track_id)
                        self.seen_track_ids.add(track_id)

                        x1, y1, x2, y2 = xyxy_coords[i]
                        confidence = float(confs[i])
                        bbox_tuple = (float(x1), float(y1), float(x2), float(y2))

                        # Bottom-center coordinate: (x_center, y_feet) on the floor plane
                        x_center = float((x1 + x2) / 2.0)
                        y_feet = float(y2)
                        curr_point = (x_center, y_feet)
                        curr_coord = x_center if clean_orientation == "vertical" else y_feet

                        # Cache current detection and active track bounding box
                        self.cached_detections.append({
                            "track_id": track_id,
                            "bbox": bbox_tuple,
                            "confidence": confidence,
                            "feet_point": curr_point,
                        })
                        self.cached_active_boxes[track_id] = bbox_tuple

                        if track_id not in self.tracks:
                            track_info = TrackInfo(
                                track_id=track_id,
                                state=TrackState.IDLE,
                                last_seen_frame=current_frame,
                                prev_feet_point=curr_point,
                                last_feet_point=curr_point,
                                bbox=bbox_tuple,
                                confidence=confidence,
                                velocity=(0.0, 0.0),
                            )
                            track_info.trajectory.append(curr_point)
                            self.tracks[track_id] = track_info
                            prev_coord = None
                        else:
                            track_info = self.tracks[track_id]
                            if track_info.last_feet_point is not None:
                                prev_coord = (
                                    track_info.last_feet_point[0]
                                    if clean_orientation == "vertical"
                                    else track_info.last_feet_point[1]
                                )
                            else:
                                prev_coord = None

                            track_info.prev_feet_point = track_info.last_feet_point
                            track_info.last_feet_point = curr_point
                            track_info.last_seen_frame = current_frame
                            track_info.bbox = bbox_tuple
                            track_info.confidence = confidence
                            track_info.trajectory.append(curr_point)

                        # Update FSM Crossing Logic
                        event_type = self._update_fsm(
                            track_id=track_id,
                            prev_coord=prev_coord,
                            curr_coord=curr_coord,
                            line_a=line_a_pixel,
                            line_b=line_b_pixel,
                            delta=delta_y,
                            timeout_sec=timeout_sec,
                            current_time=current_time,
                            current_frame=current_frame,
                            orientation=clean_orientation,
                        )
                        if event_type is not None:
                            new_events.append(self.events[-1])

                        current_frame_tracks.append(track_info)

            # Update confirmed active tracks: If ByteTrack does not return a track ID in the current
            # detection frame, remove it from active rendering immediately.
            self.active_track_ids = set(current_detected_ids)
            self.last_detection_frame = current_frame

        else:
            # Intermediate / skipped frame (when skip_detection is True):
            # No velocity extrapolation logic to prevent ghost boxes or floating artifacts.
            if self.last_detection_frame is not None:
                active_ids = [t_id for t_id in self.active_track_ids if t_id in self.tracks]
            else:
                active_ids = [t_id for t_id in self.cached_active_boxes if t_id in self.tracks] or list(self.tracks.keys())

            for track_id in active_ids:
                track_info = self.tracks[track_id]
                if track_info.last_feet_point is None:
                    continue

                curr_coord = (
                    track_info.last_feet_point[0]
                    if clean_orientation == "vertical"
                    else track_info.last_feet_point[1]
                )
                if track_info.prev_feet_point:
                    prev_coord = (
                        track_info.prev_feet_point[0]
                        if clean_orientation == "vertical"
                        else track_info.prev_feet_point[1]
                    )
                else:
                    prev_coord = curr_coord

                track_info.last_seen_frame = current_frame

                event_type = self._update_fsm(
                    track_id=track_id,
                    prev_coord=prev_coord,
                    curr_coord=curr_coord,
                    line_a=line_a_pixel,
                    line_b=line_b_pixel,
                    delta=delta_y,
                    timeout_sec=timeout_sec,
                    current_time=current_time,
                    current_frame=current_frame,
                    orientation=clean_orientation,
                )
                if event_type is not None:
                    new_events.append(self.events[-1])

                current_frame_tracks.append(track_info)

        # 3. Memory Garbage Collection: purge tracks lost for > purge_timeout_frames
        self._purge_stale_tracks(current_frame)

        # 4. Generate Visual Overlays
        try:
            annotated_frame = self._draw_overlays(
                frame=frame,
                line_a=line_a_pixel,
                line_b=line_b_pixel,
                current_frame_tracks=current_frame_tracks,
                orientation=clean_orientation,
            )
        except Exception as exc:
            logger.warning(
                "Frame %d: Visual overlay rendering anomaly caught: %s. Using clean frame fallback.",
                current_frame,
                exc,
            )
            annotated_frame = frame.copy()

        frame_summary = {
            "frame": current_frame,
            "timestamp": current_time,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "occupancy": self.occupancy,
            "active_tracks": len(self.tracks),
            "new_events": new_events,
            "skipped_detection": skip_detection,
            "orientation": clean_orientation,
        }

        return annotated_frame, frame_summary

    def process_video(
        self,
        input_path: str,
        output_raw_path: Optional[str] = None,
        line_a_norm: float = 0.45,
        line_b_norm: float = 0.55,
        timeout_sec: float = 4.0,
        delta_y: float = 10.0,
        frame_stride: int = 1,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        orientation: str = "horizontal",
    ) -> Dict[str, Any]:
        """Process an entire video file, writing annotated output if desired and returning metrics.

        Args:
            input_path: Absolute or relative path to the input video file.
            output_raw_path: Optional path to write annotated video output (.mp4, .avi).
            line_a_norm: Normalized coordinate ratio for Line A (0.0 to 1.0).
            line_b_norm: Normalized coordinate ratio for Line B (0.0 to 1.0).
            timeout_sec: Timeout for incomplete crossings in seconds (default 4.0s).
            delta_y: Hysteresis deadband margin in pixels.
            frame_stride: Inference stride (defaults to 1 for full per-frame processing).
            progress_callback: Optional callback func(current_frame, total_frames, current_fps).
            orientation: "horizontal" (Top/Bottom flow) or "vertical" (Left/Right flow).

        Returns:
            Dictionary containing:
            {
                "total_in": int,
                "total_out": int,
                "occupancy": int,
                "total_tracks": int,
                "events": [{"frame": int, "time_sec": float, "id": int, "type": "IN"|"OUT"}],
                "avg_fps": float,
                "total_frames": int,
                "orientation": str
            }
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found at '{input_path}'")

        # Reset state for fresh video run
        self.reset()

        clean_orientation = orientation.lower().strip() if orientation else "horizontal"
        if clean_orientation != "vertical":
            clean_orientation = "horizontal"

        cap: Optional[cv2.VideoCapture] = None
        writer: Optional[cv2.VideoWriter] = None

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source at '{input_path}'")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or np.isnan(fps):
                fps = 30.0  # Fallback default FPS

            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Resolution scaling optimization:
            # If native width exceeds 1280 (e.g., 1080p or 4K surveillance video):
            # Calculate a scaled dimension maintaining exact aspect ratio with target width = 960:
            # scale = 960.0 / orig_w
            # target_w = 960
            # target_h = int(round(orig_h * scale))
            # Reduces inference overhead, encoding latency, and disk I/O by >60%.
            if orig_w > 1280:
                scale = 960.0 / float(orig_w)
                target_w = 960
                target_h = int(round(orig_h * scale))
                do_resize = True
                logger.info(
                    "Native video resolution (%dx%d) exceeds 1280px. Downscaling to %dx%d (scale=%.4f).",
                    orig_w,
                    orig_h,
                    target_w,
                    target_h,
                    scale,
                )
            else:
                target_w = orig_w
                target_h = orig_h
                do_resize = False

            width = target_w
            height = target_h

            logger.info(
                "Opened video '%s' [%dx%d @ %.2f FPS, %d total frames, frame_stride=%d, orientation=%s]",
                input_path,
                width,
                height,
                fps,
                total_frames,
                frame_stride,
                clean_orientation,
            )

            if output_raw_path:
                out_dir = os.path.dirname(output_raw_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)

                # Select codec: mp4v is standard across platforms
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_raw_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    # Fallback to XVID if mp4v fails
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(output_raw_path, fourcc, fps, (width, height))

                logger.info("Writing annotated output to '%s' [%dx%d]", output_raw_path, width, height)

            processed_frames = 0
            start_wall_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if do_resize:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                frame_idx = processed_frames
                frame_time_sec = frame_idx / fps

                should_skip = (frame_stride > 1) and (frame_idx % frame_stride != 0)

                try:
                    annotated_frame, _ = self.process_frame(
                        frame=frame,
                        line_a_norm=line_a_norm,
                        line_b_norm=line_b_norm,
                        timeout_sec=timeout_sec,
                        delta_y=delta_y,
                        frame_idx=frame_idx + 1,
                        timestamp=frame_time_sec,
                        skip_detection=should_skip,
                        orientation=clean_orientation,
                    )
                except Exception as frame_exc:
                    logger.warning(
                        "Frame %d: Per-frame processing/drawing anomaly caught: %s. Continuing with unannotated frame.",
                        frame_idx + 1,
                        frame_exc,
                    )
                    annotated_frame = frame.copy()

                if writer is not None:
                    writer.write(annotated_frame)

                processed_frames += 1

                if progress_callback is not None:
                    elapsed = max(0.001, time.time() - start_wall_time)
                    current_fps = processed_frames / elapsed
                    progress_callback(processed_frames, total_frames, current_fps)

        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

        total_elapsed = max(0.001, time.time() - start_wall_time)
        avg_fps = float(processed_frames / total_elapsed)

        logger.info(
            "Video processing finished. Processed %d frames in %.2fs (Avg FPS: %.2f). IN: %d, OUT: %d (Orientation: %s)",
            processed_frames,
            total_elapsed,
            avg_fps,
            self.total_in,
            self.total_out,
            clean_orientation,
        )

        return {
            "total_in": int(self.total_in),
            "total_out": int(self.total_out),
            "occupancy": int(self.occupancy),
            "total_tracks": int(len(self.seen_track_ids)),
            "events": list(self.events),
            "avg_fps": float(round(avg_fps, 2)),
            "total_frames": int(processed_frames),
            "orientation": clean_orientation,
        }
