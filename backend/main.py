"""backend/main.py - FastAPI Service for ProGlint Footfall Intelligence.

What is where:
- RUNS_DIR & app: FastAPI application setup with CORS middleware.
- engine: Global FootfallEngine instance.
- to_h264(): Transcodes raw OpenCV video to web-streamable H.264 MP4 using FFmpeg.
- GET /health: Health check returning device, detector, and tracker info.
- POST /process_video: Uploads video, runs FootfallEngine, transcodes, returns metrics.
- GET /videos/{filename}: Streams processed video file to browser/frontend.
"""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] os for file paths, uuid for unique task IDs, subprocess for FFmpeg
import os, uuid, subprocess
# [FROM: PyTorch] torch to detect CUDA GPU acceleration
import torch
# [FROM: fastapi] FastAPI framework components for REST endpoints, file uploads, exceptions
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# [FROM: engine.tracker] Imports the core FootfallEngine from our engine package
from engine.tracker import FootfallEngine

# ==============================================================================
# 2. SERVICE SETUP & INITIALIZATION
# ==============================================================================
# [DEF & INIT: RUNS_DIR] Directory for saving temporary uploads and processed video files
# [USED IN: /process_video, /videos/{filename}]
RUNS_DIR = os.path.abspath("runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# [DEF & INIT: FastAPI App] Main REST API application instance
# [USED IN: run.py to launch backend via Uvicorn]
app = FastAPI(title="ProGlint Core API")

# [FUNCTION: CORS Middleware] Allows Streamlit frontend (port 8501) to talk to FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# [INIT: Global Engine] Initializes single FootfallEngine instance in RAM at backend startup
# [FROM: engine.tracker.FootfallEngine]
engine = FootfallEngine()

# ==============================================================================
# 3. VIDEO TRANSCODING HELPER
# ==============================================================================
# [DEF: to_h264()] Transcodes raw OpenCV video to browser-compatible H.264 MP4
# [FUNCTION: FFmpeg subprocess] Uses libx264 and yuv420p pixel format required by HTML5 video
# [USED IN: Called inside /process_video after OpenCV finishes writing]
def to_h264(src: str, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", dst],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# ==============================================================================
# 4. REST API ENDPOINTS
# ==============================================================================
# [DEF: GET /health] System health and hardware diagnostic endpoint
# [USED IN: Streamlit frontend, automated monitoring, tests/test_backend_api.py]
@app.get("/health")
def health():
    hw = "CUDA (RTX 4060)" if torch.cuda.is_available() and "cuda" in engine.device else "CPU"
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": engine.device,
        "detector": "Domain-Adapted RT-DETR (PETS-2009)",
        "tracker": "BoT-SORT",
        "hardware": hw
    }

# [DEF: POST /process_video] Primary video inference endpoint
# [FUNCTION: Pipeline Execution] Saves uploaded file -> runs engine -> transcodes -> returns JSON
# [USED IN: Called by frontend/app.py when user clicks 'Start Footfall Analysis']
@app.post("/process_video")
async def process_video(
    video: UploadFile = File(...),         # Uploaded video file from frontend
    line_a_norm: float = Form(0.45),      # Normalized Y for Line A (0.0 to 1.0)
    line_b_norm: float = Form(0.55),      # Normalized Y for Line B (0.0 to 1.0)
    timeout_sec: float = Form(3.0)        # FSM lingering timeout in seconds
):
    # Validate allowed video extension
    orig_ext = os.path.splitext(video.filename or "")[1].lower()
    if orig_ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    # Generate unique task filename to prevent collisions between requests
    uid = uuid.uuid4().hex[:8]
    tmp_in = os.path.join(RUNS_DIR, f"in_{uid}{orig_ext}")
    raw_out = os.path.join(RUNS_DIR, f"raw_{uid}.mp4")
    final_mp4 = f"{uid}_processed.mp4"
    final_path = os.path.join(RUNS_DIR, final_mp4)

    # Save uploaded bytes to disk
    with open(tmp_in, "wb") as f:
        f.write(await video.read())

    # [FUNCTION: Call Engine] Runs the core tracking and counting pipeline
    # [FROM: engine.tracker.FootfallEngine.process_video]
    res = engine.process_video(tmp_in, raw_out, line_a_norm, line_b_norm, timeout_sec)
    
    # Transcode OpenCV raw mp4v to web-friendly H.264
    to_h264(raw_out, final_path)

    # Clean up temporary raw files
    for f in (tmp_in, raw_out):
        if os.path.exists(f): os.remove(f)

    # Return structured JSON response to frontend
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

# [DEF: GET /videos/{filename}] Video streaming endpoint
# [FUNCTION: File Streaming] Returns MP4 video bytes with proper MIME type for browser playback
# [USED IN: Called by frontend/app.py st.video() to play the processed output]
@app.get("/videos/{filename}")
def get_video(filename: str):
    path = os.path.join(RUNS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")
