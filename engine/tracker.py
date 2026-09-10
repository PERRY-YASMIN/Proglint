"""engine/tracker.py - Core Footfall Engine (Detection + Tracking + Directional FSM).

Architecture Overview:
- Detection: RT-DETR (Vision Transformer) from Ultralytics
- Tracking: BoT-SORT (Kalman Filter + Re-ID feature association)
- Counting: Dual-tripwire Finite State Machine (FSM) using foot-point tracking
"""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] os, time for file paths and timestamp calculations
import os, time
# [FROM: opencv-python] cv2 for image processing, drawing lines/boxes, video capture/writing
import cv2
# [FROM: PyTorch] torch to detect NVIDIA GPU / CUDA acceleration
import torch
# [FROM: NumPy] np for bounding box coordinate arrays and matrix operations
import numpy as np
# [FROM: typing] Type annotations for cleaner function signatures
from typing import Any, Dict, Tuple
# [FROM: ultralytics] RT-DETR Vision Transformer object detection model
from ultralytics import RTDETR

# ==============================================================================
# 2. MODEL PATH CONFIGURATION
# ==============================================================================
# [DEF: DEFAULT_MODEL] Path to base pre-trained RT-DETR-L weights
# [USED IN: FootfallEngine.__init__(), tests/test_footfall_engine.py]
DEFAULT_MODEL = "rtdetr-l.pt"

# [DEF: CUSTOM_WEIGHTS_PATH] Path where domain-adapted fine-tuned weights are saved
# [USED IN: evaluation/benchmark.py, tests/test_footfall_engine.py]
CUSTOM_WEIGHTS_PATH = "runs/custom_train/best_rtdetr.pt"

# [DEF: get_default_model_path()] Helper returning default model weight path
# [USED IN: tests/test_footfall_engine.py to verify model path contract]
def get_default_model_path() -> str:
    """Returns the default model weight path (rtdetr-l.pt)."""
    return DEFAULT_MODEL

# ==============================================================================
# 3. CORE FOOTFALL ENGINE CLASS
# ==============================================================================
# [DEF: FootfallEngine] Main class containing detection, tracking, and counting state
# [USED IN: backend/main.py, evaluation/benchmark.py, tests/test_footfall_engine.py]
class FootfallEngine:
    def __init__(self, model_path: str = DEFAULT_MODEL, conf_threshold: float = 0.40):
        # [INIT: Device Selection] Automatically locks onto CUDA GPU if available, else CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # [INIT: Model Loading] Loads the RT-DETR Vision Transformer checkpoint into memory
        # [FROM: ultralytics.RTDETR]
        self.model = RTDETR(model_path)
        self.conf = conf_threshold
        
        # [INIT: State Counters] Initializes total counts and tracking history dictionaries
        self.reset()

    # [DEF: reset()] Resets all counters, logs, and tracking memory
    # [USED IN: Called at startup and before processing each new video file]
    def reset(self):
        # [INIT: Counters] Total people entered and exited
        self.total_in = 0
        self.total_out = 0
        # [INIT: Event Log] Stores timestamps and track IDs for each crossing
        self.events = []
        # [INIT: Unique IDs] Set of all distinct track IDs observed across the session
        self.seen_ids = set()
        # [INIT: History Table] track_id -> [last_y, state, t_start, frames_absent, start_y]
        self.history: Dict[int, list] = {}

    # [DEF: occupancy] Net current occupancy inside the monitored zone
    # [FUNCTION: Logic] Mathematical formula: net occupancy = total_in - total_out
    # [USED IN: backend/main.py, frontend/app.py, evaluation/benchmark.py]
    @property
    def occupancy(self) -> int:
        return self.total_in - self.total_out

    # [DEF: process_frame()] Processes a single video frame through the full pipeline
    # [FUNCTION: Workflow] Detection -> BoT-SORT Tracking -> Foot Point -> FSM -> HUD Overlay
    # [USED IN: backend/main.py, process_video(), tests/test_footfall_engine.py]
    def process_frame(self, frame: np.ndarray, line_a: float = 0.45, line_b: float = 0.55,
                      timeout: float = 3.0, frame_idx: int = 0, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        # [FUNCTION: Frame Dimensions] Extracts frame height (h) and width (w) in pixels
        h, w = frame.shape[:2]
        
        # [FUNCTION: Line Conversion] Converts normalized positions (0.0 to 1.0) to pixel Y coordinates
        # [FROM: kwargs sliders from frontend/app.py or defaults line_a/line_b]
        y_a = int(kwargs.get("line_a_norm", line_a) * h)  # Top tripwire (Line A)
        y_b = int(kwargs.get("line_b_norm", line_b) * h)  # Bottom tripwire (Line B)
        now = time.time()
        annotated = frame.copy()

        # [INIT & FUNCTION: Detection + Tracking]
        # Runs RT-DETR detection for class 0 (person) + BoT-SORT multi-object tracking
        # [FROM: ultralytics model.track()]
        res = self.model.track(
            source=frame,
            persist=True,          # Preserves track IDs between consecutive frames
            classes=[0],           # 0 = Person class in COCO / PETS-2009
            conf=self.conf,        # Confidence threshold (0.40)
            tracker="botsort.yaml",# BoT-SORT tracker config (Kalman + Re-ID)
            imgsz=640,             # Standard resolution for real-time inference
            device=self.device,    # Runs on CUDA GPU
            verbose=False
        )

        active_ids = set()
        # Check if any bounding boxes with track IDs were detected in this frame
        if res and res[0].boxes and res[0].boxes.id is not None:
            # [FROM: PyTorch Tensors -> NumPy Arrays]
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, ids):
                if np.any(np.isnan(box)): continue
                active_ids.add(tid)
                self.seen_ids.add(tid)
                
                # [FUNCTION: Bounding Box Coordinates]
                x1, y1, x2, y2 = [int(round(float(c))) for c in box]
                
                # [FUNCTION: Foot-Point Anchor]
                # In overhead CCTV, the torso shifts with height. We track (cx, y2)
                # which is the bottom-center point where feet contact the ground plane.
                cx, cy = int((x1 + x2) / 2), y2

                # [FUNCTION: Retrieve Track State]
                # Fetches (last_y, state, t_start, frames_absent, start_y) or initializes new track
                prev_y, state, t_start, _, start_y = self.history.get(tid, [cy, "IDLE", now, 0, cy])
                
                # [FUNCTION: Temporal Timeout]
                # If pedestrian lingers inside deadband longer than timeout (3s), reset to IDLE
                if (now - t_start) > timeout and state in ("PENDING_IN", "PENDING_OUT"):
                    state = "IDLE"

                # [FUNCTION: 15-Pixel Motion Filter]
                # Prevents double-counting from pedestrians swaying in place without walking
                has_moved = abs(cy - start_y) > 15

                # --------------------------------------------------------------
                # DUAL-TRIPWIRE DIRECTIONAL FINITE STATE MACHINE (FSM)
                # --------------------------------------------------------------
                # [FUNCTION: State Transitions]
                # Direction 1: Downward Crossing (Line A -> Line B = IN)
                if state == "IDLE":
                    if prev_y < y_a <= cy:
                        state, t_start = "PENDING_IN", now   # Crossed Line A going down
                    elif prev_y > y_b >= cy:
                        state, t_start = "PENDING_OUT", now  # Crossed Line B going up
                
                # Pedestrian was PENDING_IN and now crosses Line B downward -> IN count
                elif state == "PENDING_IN" and prev_y < y_b <= cy:
                    if has_moved:
                        self.total_in += 1
                        # [FUNCTION: Event Logging] Records event for Plotly timeline in frontend
                        self.events.append({"frame": frame_idx, "id": tid, "type": "IN", "time_sec": round(now, 2)})
                    state = "DONE"  # Latch into DONE to prevent duplicate counting
                
                # Pedestrian was PENDING_OUT and now crosses Line A upward -> OUT count
                elif state == "PENDING_OUT" and prev_y > y_a >= cy:
                    if has_moved:
                        self.total_out += 1
                        self.events.append({"frame": frame_idx, "id": tid, "type": "OUT", "time_sec": round(now, 2)})
                    state = "DONE"  # Latch into DONE to prevent duplicate counting

                # [FUNCTION: Update Track History]
                self.history[tid] = [cy, state, t_start, 0, start_y]

                # [FUNCTION: Visual Overlays] Draws green bounding box, ID badge, and foot point
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{tid}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        # [FUNCTION: Memory Garbage Collection]
        # Prunes old tracks absent for > 60 frames (2.4 seconds) to prevent memory leaks
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                self.history[tid][3] += 1
                if self.history[tid][3] > 60:
                    del self.history[tid]

        # [FUNCTION: Draw Tripwires & HUD Banner]
        # Cyan Line A (top), Magenta Line B (bottom), and live counting HUD
        cv2.line(annotated, (0, y_a), (w, y_a), (255, 255, 0), 2)
        cv2.line(annotated, (0, y_b), (w, y_b), (255, 0, 255), 2)
        cv2.putText(annotated, f"IN: {self.total_in} | OUT: {self.total_out} | OCCUPANCY: {self.occupancy}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return annotated, {
            "frame": frame_idx, "total_in": self.total_in, "total_out": self.total_out,
            "occupancy": self.occupancy, "active_tracks": len(active_ids)
        }

    # [DEF: process_video()] Batch processes an entire video file
    # [USED IN: backend/main.py, evaluation/benchmark.py, CLI direct execution]
    def process_video(self, input_path: str, output_raw_path: str = None, line_a_norm: float = 0.45,
                      line_b_norm: float = 0.55, timeout_sec: float = 3.0, **kwargs):
        self.reset()
        # [INIT: VideoCapture] Opens the video file for decoding via OpenCV
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise IOError(f"Cannot open video: {input_path}")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # [INIT: VideoWriter] Optional writer to save annotated output video
        writer = cv2.VideoWriter(output_raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if output_raw_path else None

        f_idx, t0 = 0, time.perf_counter()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None: break
            
            # Process each frame sequentially
            annotated, _ = self.process_frame(
                frame, line_a=line_a_norm, line_b=line_b_norm,
                timeout=timeout_sec, frame_idx=f_idx
            )
            if writer: writer.write(annotated)
            f_idx += 1

        cap.release()
        if writer: writer.release()
        elapsed = time.perf_counter() - t0
        
        # [FUNCTION: Summary Metrics] Returns full analytics package to caller
        return {
            "total_in": self.total_in, "total_out": self.total_out, "occupancy": self.occupancy,
            "total_tracks": len(self.seen_ids), "avg_fps": round(f_idx / max(elapsed, 0.001), 2),
            "total_frames": f_idx, "events": self.events
        }

# ==============================================================================
# 4. DIRECT CLI EXECUTION ENTRYPOINT
# ==============================================================================
# [FUNCTION: Direct Runner] Runs standalone when executed via: python engine/tracker.py
if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "TownCentre_test.mp4"
    out_video = os.path.join("runs", "output_counted.mp4")
    os.makedirs("runs", exist_ok=True)
    print(f"[ProGlint] Running Footfall Engine directly on: {video}")
    engine = FootfallEngine()
    stats = engine.process_video(video, output_raw_path=out_video)
    print(f"[Done] Total Frames: {stats['total_frames']} | Avg FPS: {stats['avg_fps']}")
    print(f"[Result] IN: {stats['total_in']} | OUT: {stats['total_out']} | Net Occupancy: {stats['occupancy']}")
    print(f"[Saved] Output video saved to: {out_video}")
