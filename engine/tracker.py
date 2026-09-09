"""engine/tracker.py - Unified Footfall Detection, BoT-SORT Tracking, and Line-Crossing Engine."""
import os, time, cv2, torch, numpy as np
from typing import Any, Dict, Optional, Tuple
from ultralytics import RTDETR

# Use base pre-trained RT-DETR to avoid overfitted artifacts
DEFAULT_MODEL = "rtdetr-l.pt"
CUSTOM_WEIGHTS_PATH = "runs/custom_train/best_rtdetr.pt"

def get_default_model_path() -> str:
    return DEFAULT_MODEL

class FootfallEngine:
    def __init__(self, model_path: str = DEFAULT_MODEL, conf_threshold: float = 0.40):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = RTDETR(model_path)
        self.conf = conf_threshold
        self.reset()

    def reset(self):
        self.total_in = 0
        self.total_out = 0
        self.events = []
        self.seen_ids = set()
        self.history: Dict[int, list] = {}  # tid -> [last_y, state, t_start, frames_absent, start_y]

    @property
    def occupancy(self) -> int:
        return self.total_in - self.total_out

    def process_frame(self, frame: np.ndarray, line_a: float = 0.45, line_b: float = 0.55,
                      timeout: float = 3.0, frame_idx: int = 0, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = frame.shape[:2]
        line_a_norm = kwargs.get("line_a_norm", line_a)
        line_b_norm = kwargs.get("line_b_norm", line_b)
        y_a, y_b = int(line_a_norm * h), int(line_b_norm * h)
        now = time.time()
        annotated = frame.copy()

        # Fixed imgsz=640 and conf=0.40
        res = self.model.track(source=frame, persist=True, classes=[0], conf=self.conf,
                               tracker="botsort.yaml", imgsz=640, device=self.device, verbose=False)

        active_ids = set()
        if res and res[0].boxes and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, ids):
                if np.any(np.isnan(box)):
                    continue
                active_ids.add(tid)
                self.seen_ids.add(tid)
                x1, y1, x2, y2 = [int(round(float(c))) for c in box]
                cx, cy = int((x1 + x2) / 2), y2

                prev_y, state, t_start, _, start_y = self.history.get(tid, [cy, "IDLE", now, 0, cy])
                if (now - t_start) > timeout and state in ("PENDING_IN", "PENDING_OUT"):
                    state = "IDLE"

                # Motion filter: Ignore stationary objects with < 15px total displacement
                has_moved = abs(cy - start_y) > 15

                # Directional Crossing FSM
                if state == "IDLE":
                    if prev_y < y_a <= cy:
                        state, t_start = "PENDING_IN", now
                    elif prev_y > y_b >= cy:
                        state, t_start = "PENDING_OUT", now
                elif state == "PENDING_IN" and prev_y < y_b <= cy:
                    if has_moved:
                        self.total_in += 1
                        self.events.append({"frame": frame_idx, "id": tid, "type": "IN", "time_sec": round(now, 2)})
                    state = "DONE"
                elif state == "PENDING_OUT" and prev_y > y_a >= cy:
                    if has_moved:
                        self.total_out += 1
                        self.events.append({"frame": frame_idx, "id": tid, "type": "OUT", "time_sec": round(now, 2)})
                    state = "DONE"

                self.history[tid] = [cy, state, t_start, 0, start_y]

                # Render bounding box and ID
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{tid}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        # Garbage collect inactive tracks
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                self.history[tid][3] += 1
                if self.history[tid][3] > 60:
                    del self.history[tid]

        # Draw tripwires and HUD
        cv2.line(annotated, (0, y_a), (w, y_a), (255, 255, 0), 2)
        cv2.line(annotated, (0, y_b), (w, y_b), (255, 0, 255), 2)
        cv2.putText(annotated, f"IN: {self.total_in} | OUT: {self.total_out} | OCCUPANCY: {self.occupancy}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        summary = {"frame": frame_idx, "total_in": self.total_in, "total_out": self.total_out,
                   "occupancy": self.occupancy, "active_tracks": len(active_ids)}
        return annotated, summary

    def process_video(self, input_path: str, output_raw_path: str, line_a_norm: float = 0.45,
                      line_b_norm: float = 0.55, timeout_sec: float = 3.0, **kwargs):
        self.reset()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(output_raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if output_raw_path else None

        f_idx, t0 = 0, time.perf_counter()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            annotated, _ = self.process_frame(frame, line_a=line_a_norm, line_b=line_b_norm,
                                              timeout=timeout_sec, frame_idx=f_idx)
            if writer:
                writer.write(annotated)
            f_idx += 1

        cap.release()
        if writer:
            writer.release()
        elapsed = time.perf_counter() - t0
        return {
            "total_in": self.total_in, "total_out": self.total_out, "occupancy": self.occupancy,
            "total_tracks": len(self.seen_ids), "avg_fps": round(f_idx / max(elapsed, 0.001), 2),
            "total_frames": f_idx, "events": self.events
        }