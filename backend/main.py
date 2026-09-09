"""backend/main.py - FastAPI Service for ProGlint Footfall Intelligence."""
import os, uuid, subprocess, torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from engine.tracker import FootfallEngine

RUNS_DIR = os.path.abspath("runs")
os.makedirs(RUNS_DIR, exist_ok=True)

app = FastAPI(title="ProGlint Core API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
engine = FootfallEngine()

def to_h264(src, dst):
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", dst],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@app.get("/health")
def health():
    hw = "CUDA (RTX 4060)" if torch.cuda.is_available() and "cuda" in engine.device else "CPU"
    return {"status": "ok", "cuda_available": torch.cuda.is_available(), "device": engine.device,
            "detector": "Domain-Adapted RT-DETR (PETS-2009)", "tracker": "BoT-SORT", "hardware": hw}

@app.post("/process_video")
async def process_video(video: UploadFile = File(...), line_a_norm: float = Form(0.45),
                        line_b_norm: float = Form(0.55), timeout_sec: float = Form(3.0)):
    orig_ext = os.path.splitext(video.filename or "")[1].lower()
    if orig_ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    uid = uuid.uuid4().hex[:8]
    tmp_in = os.path.join(RUNS_DIR, f"in_{uid}{orig_ext}")
    raw_out = os.path.join(RUNS_DIR, f"raw_{uid}.mp4")
    final_mp4 = f"{uid}_processed.mp4"
    final_path = os.path.join(RUNS_DIR, final_mp4)

    with open(tmp_in, "wb") as f:
        f.write(await video.read())

    res = engine.process_video(tmp_in, raw_out, line_a_norm, line_b_norm, timeout_sec)
    to_h264(raw_out, final_path)

    for f in (tmp_in, raw_out):
        if os.path.exists(f): os.remove(f)

    # Standardized contract matching the frontend expectations
    return {
        "status": "success",
        "video_filename": final_mp4,
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
        "events": res["events"]
    }

@app.get("/videos/{filename}")
def get_video(filename: str):
    path = os.path.join(RUNS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")