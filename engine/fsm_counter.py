"""
fsm_counter.py - Dual-Tripwire Finite State Machine for Directional Footfall Counting.
Handles directional crossings (IN / OUT), spatial deadband, and double-count prevention.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional, Tuple


class TrackState(str, Enum):
    IDLE = "IDLE"
    PENDING_IN = "PENDING_IN"
    PENDING_OUT = "PENDING_OUT"
    COMPLETED = "COMPLETED"


@dataclass
class TrackInfo:
    """Stores spatial trajectory and state for an active pedestrian track ID."""
    track_id: int
    state: TrackState = TrackState.IDLE
    state_start_time: float = 0.0
    last_seen_frame: int = 0
    prev_feet_point: Optional[Tuple[float, float]] = None
    last_feet_point: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))


class DualTripwireFSM:
    """Directional crossing FSM. Requires crossing Line A then Line B (or vice versa)."""

    def __init__(self, hysteresis_delta: float = 10.0, timeout_sec: float = 4.0) -> None:
        self.hysteresis_delta = float(hysteresis_delta)
        self.timeout_sec = float(timeout_sec)

    def update_fsm(
        self,
        track: TrackInfo,
        prev_coord: Optional[float],
        curr_coord: Optional[float],
        line_a: Optional[int],
        line_b: Optional[int],
        delta: Optional[float] = None,
        timeout_sec: Optional[float] = None,
        current_time: float = 0.0,
        current_frame: int = 0,
        orientation: str = "horizontal",
        axis: Optional[str] = None,
    ) -> Optional[str]:
        if prev_coord is None or curr_coord is None or line_a is None or line_b is None:
            return None

        # Lock completed tracks to prevent double counting
        if track.state == TrackState.COMPLETED:
            return None

        now = current_time or time.time()
        timeout = timeout_sec or self.timeout_sec

        # Reset state if pedestrian lingers too long in the gate
        if track.state in (TrackState.PENDING_IN, TrackState.PENDING_OUT):
            if (now - track.state_start_time) > timeout:
                track.state = TrackState.IDLE

        # Downward / Rightward motion: Line A -> Line B = IN
        if line_a < line_b:
            if track.state == TrackState.IDLE:
                if prev_coord <= line_a and curr_coord > line_a:
                    track.state = TrackState.PENDING_IN
                    track.state_start_time = now
                elif prev_coord >= line_b and curr_coord < line_b:
                    track.state = TrackState.PENDING_OUT
                    track.state_start_time = now

            elif track.state == TrackState.PENDING_IN:
                if prev_coord < line_b and curr_coord >= line_b:
                    track.state = TrackState.COMPLETED
                    return "IN"
                elif curr_coord < line_a:  # Turned around
                    track.state = TrackState.IDLE

            elif track.state == TrackState.PENDING_OUT:
                if prev_coord > line_a and curr_coord <= line_a:
                    track.state = TrackState.COMPLETED
                    return "OUT"
                elif curr_coord > line_b:  # Turned around
                    track.state = TrackState.IDLE

        return None
