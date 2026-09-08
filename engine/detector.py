"""
detector.py - RT-DETR Vision Transformer detector interface.
"""

import os
from typing import Any, Optional
import torch
from ultralytics import RTDETR

DEFAULT_MODEL_PATH = "runs/custom_train/best_rtdetr.pt"


class RTDETRDetector:
    """Wraps Ultralytics RT-DETR for pedestrian detection."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: Optional[str] = None):
        if not os.path.exists(model_path) and not ("rtdetr" in model_path.lower() and not model_path.endswith(".pt")):
            raise FileNotFoundError(f"RT-DETR weights not found at: {model_path}")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RTDETR(model_path)

    def track(self, frame: Any, tracker: str = "botsort.yaml", **kwargs):
        use_half = bool(torch.cuda.is_available() and "cuda" in str(self.device).lower())
        return self.model.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker=tracker,
            device=self.device,
            half=use_half,
            verbose=False,
            **kwargs,
        )


def get_detector(model_path: str = DEFAULT_MODEL_PATH, device: Optional[str] = None):
    return RTDETRDetector(model_path=model_path, device=device)
