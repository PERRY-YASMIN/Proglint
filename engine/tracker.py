"""
tracker.py - Real-Time Footfall Tracking Engine.
Uses RT-DETR Vision Transformer + BoT-SORT + Dual Virtual Tripwires.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from ultralytics import RTDETR

from engine.fsm_counter import DualTripwireFSM, TrackInfo, TrackState

logger = logging.getLogger("FootfallEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

RTDETR_WEIGHTS_PATH: str = "runs/custom_train/best_rtdetr.pt"
CUSTOM_WEIGHTS_PATH: str = RTDETR_WEIGHTS_PATH
DEFAULT_MODEL_PATH: str = RTDETR_WEIGHTS_PATH


def get_default_model_path() -> str:
    """Return default model path or raise error if missing."""
    if os.path.exists(RTDETR_WEIGHTS_PATH):
        return RTDETR_WEIGHTS_PATH
    raise FileNotFoundError(f"RT-DETR domain-adapted weights not found at path: '{RTDETR_WEIGHTS_PATH}'")


class FootfallEngine:
    """Footfall tracking engine powered by RT-DETR, BoT-SORT, and Dual Virtual Tripwires."""

    def __init__(
        self,
        model_path: str = RTDETR_WEIGHTS_PATH,
        device: Optional[str] = None,
        conf_threshold: float = 0.30,
        tracker_config: str = "botsort.yaml",
        purge_timeout_frames: int = 60,
    ) -> None:
        # 1. Model Initialization
        self.model_path = model_path if model_path is not None else get_default_model_path()
        if not os.path.exists(self.model_path) and not ("rtdetr" in str(self.model_path).lower() and not str(self.model_path).endswith(".pt")):
            raise FileNotFoundError(f"RT-DETR domain-adapted weights not found at path: '{self.model_path}'")

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RTDETR(self.model_path)

        self.tracker_config = tracker_config
        self.conf_threshold = float(conf_threshold)
        self.purge_timeout_frames = int(purge_timeout_frames)
        self.fsm = DualTripwireFSM(hysteresis_delta=10.0, timeout_sec=4.0)

        # Counting and tracking state
        self.total_in = 0
        self.total_out = 0
        self.events: List[Dict[str, Any]] = []
        self.tracks: Dict[int, TrackInfo] = {}
        self.seen_track_ids: Set[int] = set()
        self.cached_detections: List[Dict[str, Any]] = []
        self.cached_active_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        self.current_frame = 0
        self.last_inference_latency_ms = 0.0

    @property
    def occupancy(self) -> int:
        return self.total_in - self.total_out

    def reset(self) -> None:
        """Reset all counters and active tracks."""
        self.total_in = 0
        self.total_out = 0
        self.events.clear()
        self.tracks.clear()
        self.seen_track_ids.clear()
        self.cached_detections.clear()
        self.cached_active_boxes.clear()
        self.current_frame = 0
        self.last_inference_latency_ms = 0.0

    def get_counts(self) -> Dict[str, int]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "occupancy": self.occupancy,
            "active_tracks": len(self.tracks),
            "total_unique_tracks": len(self.seen_track_ids),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        line_a_norm: float = 0.45,
        line_b_norm: float = 0.55,
        timeout_sec: float = 4.0,
        delta_y: float = 10.0,
        frame_idx: Optional[int] = None,
        skip_detection: bool = False,
        orientation: str = "horizontal",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single video frame with RT-DETR + BoT-SORT + DualTripwireFSM."""
        self.current_frame = frame_idx if frame_idx is not None else self.current_frame + 1
        h, w = frame.shape[:2]
        line_a = int(line_a_norm * h)
        line_b = int(line_b_norm * h)
        annotated = frame.copy()
        new_events = []
        inf_ms = 0.0

        if not skip_detection:
            self.cached_detections.clear()
            self.cached_active_boxes.clear()
            use_fp16 = bool(torch.cuda.is_available() and "cuda" in str(self.device).lower())

            # 2. Tracking Inference Call
            t0 = time.perf_counter()
            results = self.model.track(
                source=frame,
                persist=True,
                classes=[0],
                tracker=self.tracker_config,
                conf=self.conf_threshold,
                device=self.device,
                half=use_fp16,
                verbose=False,
            )
            inf_ms = (time.perf_counter() - t0) * 1000.0
            self.last_inference_latency_ms = inf_ms

            if results and len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for tid, box in zip(track_ids, boxes):
                    self.seen_track_ids.add(tid)
                    x1, y1, x2, y2 = map(float, box)
                    cx, cy = (x1 + x2) / 2.0, float(y2)
                    self.cached_detections.append({"track_id": tid, "bbox": (x1, y1, x2, y2), "feet_point": (cx, cy)})
                    self.cached_active_boxes[tid] = (x1, y1, x2, y2)

                    if tid not in self.tracks:
                        self.tracks[tid] = TrackInfo(
                            track_id=tid,
                            last_seen_frame=self.current_frame,
                            prev_feet_point=(cx, cy),
                            last_feet_point=(cx, cy),
                            bbox=(x1, y1, x2, y2),
                        )
                    else:
                        t_info = self.tracks[tid]
                        prev_y = t_info.last_feet_point[1] if t_info.last_feet_point else cy
                        crossing = self.fsm.update_fsm(
                            track=t_info,
                            prev_coord=prev_y,
                            curr_coord=cy,
                            line_a=line_a,
                            line_b=line_b,
                            timeout_sec=timeout_sec,
                            current_frame=self.current_frame,
                        )
                        t_info.prev_feet_point = t_info.last_feet_point
                        t_info.last_feet_point = (cx, cy)
                        t_info.bbox = (x1, y1, x2, y2)
                        t_info.last_seen_frame = self.current_frame

                        if crossing == "IN":
                            self.total_in += 1
                            ev = {"frame": self.current_frame, "id": tid, "type": "IN"}
                            self.events.append(ev)
                            new_events.append(ev)
                        elif crossing == "OUT":
                            self.total_out += 1
                            ev = {"frame": self.current_frame, "id": tid, "type": "OUT"}
                            self.events.append(ev)
                            new_events.append(ev)

        # Purge stale tracks lost for > purge_timeout_frames
        stale = [t for t, info in self.tracks.items() if (self.current_frame - info.last_seen_frame) > self.purge_timeout_frames]
        for t in stale:
            del self.tracks[t]

        # Draw overlays: tripwires and bounding boxes
        cv2.line(annotated, (0, line_a), (w, line_a), (255, 255, 0), 2)
        cv2.putText(annotated, "Line A (In)", (10, max(20, line_a - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.line(annotated, (0, line_b), (w, line_b), (255, 0, 255), 2)
        cv2.putText(annotated, "Line B (Out)", (10, max(20, line_b - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        for tid, t_info in self.tracks.items():
            if t_info.bbox:
                bx1, by1, bx2, by2 = map(int, t_info.bbox)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{tid}", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if t_info.last_feet_point:
                fcx, fcy = map(int, t_info.last_feet_point)
                cv2.circle(annotated, (fcx, fcy), 4, (0, 255, 255), -1)

        # Top HUD dashboard
        banner = f"IN: {self.total_in} | OUT: {self.total_out} | OCCUPANCY: {self.occupancy}"
        cv2.rectangle(annotated, (10, 10), (380, 45), (20, 20, 20), -1)
        cv2.putText(annotated, banner, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        summary = {
            "frame": self.current_frame,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "occupancy": self.occupancy,
            "active_tracks": len(self.tracks),
            "new_events": new_events,
            "skipped_detection": skip_detection,
            "inference_latency_ms": round(inf_ms, 2),
        }
        return annotated, summary

    def process_video(
        self,
        input_path: str,
        output_raw_path: Optional[str] = None,
        line_a_norm: float = 0.45,
        line_b_norm: float = 0.55,
        timeout_sec: float = 4.0,
        frame_stride: int = 1,
        orientation: str = "horizontal",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Process complete video file and save annotated video."""
        self.reset()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_raw_path:
            os.makedirs(os.path.dirname(output_raw_path) or ".", exist_ok=True)
            writer = cv2.VideoWriter(output_raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_idx = 0
        t_start = time.perf_counter()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            skip = bool(frame_stride > 1 and (frame_idx % frame_stride != 0))
            ann_frame, _ = self.process_frame(
                frame=frame,
                line_a_norm=line_a_norm,
                line_b_norm=line_b_norm,
                timeout_sec=timeout_sec,
                frame_idx=frame_idx,
                skip_detection=skip,
                orientation=orientation,
            )
            if writer:
                writer.write(ann_frame)
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        elapsed = time.perf_counter() - t_start
        avg_fps = round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0

        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "occupancy": self.occupancy,
            "total_tracks": len(self.seen_track_ids),
            "events": self.events,
            "avg_fps": avg_fps,
            "total_frames": frame_idx,
            "orientation": orientation,
        }
