"""
main.py - FastAPI Footfall Tracking Service powered by RT-DETR + BoT-SORT.
"""

import os
import shutil
import subprocess
import uuid
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.tracker import FootfallEngine

RUNS_DIR = os.path.abspath("runs")
os.makedirs(RUNS_DIR, exist_ok=True)

app = FastAPI(title="Footfall Analytics Service", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global engine instance
engine = FootfallEngine()


def transcode_to_h264(input_path: str, output_path: str) -> bool:
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", output_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        shutil.copy(input_path, output_path)
        return False


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    device: str
    detector: str
    tracker: str
    hardware: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    hw = "CUDA (RTX 4060)" if torch.cuda.is_available() and "cuda" in engine.device else "CPU"
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": engine.device,
        "detector": "Domain-Adapted RT-DETR (PETS-2009)",
        "tracker": "BoT-SORT",
        "hardware": hw,
    }


@app.post("/process_video")
async def process_video(
    video: UploadFile = File(...),
    line_a_norm: float = Form(0.45),
    line_b_norm: float = Form(0.55),
    timeout_sec: float = Form(4.0),
    frame_stride: int = Form(1),
    orientation: str = Form("horizontal"),
    **kwargs,
):
    orig_ext = os.path.splitext(video.filename or "")[1].lower()
    if orig_ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported file extension. Only .mp4, .avi, .mov allowed.")

    uid = uuid.uuid4().hex[:12]
    temp_in = os.path.join(RUNS_DIR, f"in_{uid}{orig_ext}")
    raw_out = os.path.join(RUNS_DIR, f"raw_{uid}.mp4")
    final_filename = f"{uid}_processed.mp4"
    final_out = os.path.join(RUNS_DIR, final_filename)

    with open(temp_in, "wb") as f:
        f.write(await video.read())

    res = engine.process_video(
        input_path=temp_in,
        output_raw_path=raw_out,
        line_a_norm=line_a_norm,
        line_b_norm=line_b_norm,
        timeout_sec=timeout_sec,
        frame_stride=frame_stride,
        orientation=orientation,
    )

    transcode_to_h264(raw_out, final_out)

    if os.path.exists(temp_in):
        os.remove(temp_in)
    if os.path.exists(raw_out):
        os.remove(raw_out)

    return {
        "status": "success",
        "video_filename": final_filename,
        "metrics": {
            "total_in": res["total_in"],
            "total_out": res["total_out"],
            "current_occupancy": res["occupancy"],
            "total_tracks": res["total_tracks"],
        },
        "performance": {
            "avg_fps": res["avg_fps"],
            "total_frames": res["total_frames"],
        },
        "events": res["events"],
    }


@app.get("/videos/{filename}")
def get_video(filename: str):
    path = os.path.join(RUNS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")
