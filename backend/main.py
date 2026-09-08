"""
Production FastAPI service for Real-Time Footfall Tracking and Analytics.
Interfaces directly with engine.tracker.FootfallEngine.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch

from engine.tracker import FootfallEngine

# Configure logging
logger = logging.getLogger("FootfallAPI")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Shared runs directory for processing and video hosting
BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


def get_ffmpeg_binary() -> str:
    """Resolve FFmpeg binary from PATH or fallback to bundled imageio_ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning("Could not resolve imageio_ffmpeg: %s. Defaulting to 'ffmpeg'", exc)
        return "ffmpeg"


# --- Pydantic Schema Definitions ---


class MetricsResponse(BaseModel):
    """Cumulative pedestrian counting metrics."""

    total_in: int = Field(..., description="Total people entered")
    total_out: int = Field(..., description="Total people exited")
    current_occupancy: int = Field(..., description="Net occupancy (total_in - total_out)")
    total_tracks: int = Field(..., description="Unique person track IDs observed")


class PerformanceResponse(BaseModel):
    """Inference and processing throughput metrics."""

    avg_fps: float = Field(..., description="Average processing frames per second")
    total_frames: int = Field(..., description="Total frames processed in video")


class EventItem(BaseModel):
    """Individual crossing event details."""

    frame: int = Field(..., description="Frame index of crossing")
    time_sec: float = Field(..., description="Video timestamp in seconds")
    id: int = Field(..., description="Tracking ID of person")
    type: str = Field(..., description="Direction type: 'IN' or 'OUT'")
    axis: Optional[str] = Field(None, description="Tripwire axis: 'Y' or 'X' in dual-axis mode")


class ProcessVideoResponse(BaseModel):
    """Response payload for video footfall processing."""

    status: str = Field("success", description="Execution status")
    video_filename: str = Field(..., description="Filename of processed H.264 video")
    metrics: MetricsResponse
    performance: PerformanceResponse
    events: List[EventItem]


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    cuda_available: bool
    device: str


# --- Lifespan Setup ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize FootfallEngine at startup and manage application lifecycle."""
    logger.info("Initializing FootfallEngine on startup...")
    # Initialize engine bound to CUDA if available
    app.state.engine = FootfallEngine(model_path="yolo11n.pt")
    logger.info(
        "FootfallEngine initialized on device: %s. Storage: %s",
        app.state.engine.device,
        RUNS_DIR,
    )
    yield
    logger.info("Shutting down FootfallEngine service...")


# --- FastAPI Application Configuration ---

app = FastAPI(
    title="Footfall Tracking & Analytics Service",
    description="Real-time person counting & directional tracking service powered by YOLO11 and ByteTrack",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (allow all origins, credentials, methods, headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> FootfallEngine:
    """Retrieve global FootfallEngine instance, initializing lazily if not present."""
    if not hasattr(app.state, "engine") or app.state.engine is None:
        logger.info("Initializing FootfallEngine lazily...")
        app.state.engine = FootfallEngine(model_path="yolo11n.pt")
    return app.state.engine


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """Return runtime service health, CUDA availability, and compute device."""
    engine = get_engine()
    return HealthResponse(
        status="ok",
        cuda_available=torch.cuda.is_available(),
        device=engine.device,
    )


@app.post(
    "/process_video",
    response_model=ProcessVideoResponse,
    summary="Upload and process video for bidirectional footfall tracking",
)
async def process_video(
    video: UploadFile = File(..., description="Video file (.mp4, .avi, .mov)"),
    line_a_norm: float = Form(0.45, ge=0.0, le=1.0, description="Normalized position of Line A"),
    line_b_norm: float = Form(0.55, ge=0.0, le=1.0, description="Normalized position of Line B"),
    timeout_sec: float = Form(4.0, gt=0.0, description="Timeout in seconds for pending crossings"),
    frame_stride: int = Form(1, ge=1, le=10, description="Inference frame stride (1 = full per-frame processing)"),
    orientation: str = Form("horizontal", description="Tripwire orientation: 'horizontal', 'vertical', or 'both'"),
    line_a_y_norm: Optional[float] = Form(None, ge=0.0, le=1.0, description="Normalized Y position of Line A_y (horizontal gate)"),
    line_b_y_norm: Optional[float] = Form(None, ge=0.0, le=1.0, description="Normalized Y position of Line B_y (horizontal gate)"),
    line_a_x_norm: Optional[float] = Form(None, ge=0.0, le=1.0, description="Normalized X position of Line A_x (vertical gate)"),
    line_b_x_norm: Optional[float] = Form(None, ge=0.0, le=1.0, description="Normalized X position of Line B_x (vertical gate)"),
) -> ProcessVideoResponse:
    """Process uploaded video: run tracking, dual tripwire FSM, HUD overlay, and transcode to H.264."""
    # 1. Enforce allowed video file extensions
    orig_filename = video.filename or ""
    ext = os.path.splitext(orig_filename)[1].lower()
    allowed_exts = {".mp4", ".avi", ".mov"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Only {allowed_exts} are supported.",
        )

    clean_orientation = orientation.strip().lower() if orientation else "horizontal"
    if clean_orientation not in ("horizontal", "vertical", "both"):
        clean_orientation = "horizontal"

    engine: FootfallEngine = get_engine()

    unique_id = uuid.uuid4().hex[:12]
    temp_input_path = os.path.join(RUNS_DIR, f"temp_input_{unique_id}{ext}")
    raw_output_path = os.path.join(RUNS_DIR, f"raw_output_{unique_id}.mp4")
    final_h264_filename = f"{unique_id}_processed.mp4"
    final_h264_path = os.path.join(RUNS_DIR, final_h264_filename)

    try:
        # 2. Stream and write uploaded video into runs/
        logger.info("Streaming uploaded file to '%s'...", temp_input_path)
        with open(temp_input_path, "wb") as buffer:
            while chunk := await video.read(1024 * 1024):  # 1MB chunk size
                buffer.write(chunk)

        # 3. Process video through FootfallEngine in threadpool
        logger.info(
            "Running FootfallEngine.process_video with Line A=%.2f, Line B=%.2f, Timeout=%.1fs, Stride=%d, Orientation=%s...",
            line_a_norm,
            line_b_norm,
            timeout_sec,
            frame_stride,
            clean_orientation,
        )
        result: Dict[str, Any] = await asyncio.to_thread(
            engine.process_video,
            input_path=temp_input_path,
            output_raw_path=raw_output_path,
            line_a_norm=line_a_norm,
            line_b_norm=line_b_norm,
            timeout_sec=timeout_sec,
            frame_stride=frame_stride,
            orientation=clean_orientation,
            line_a_y_norm=line_a_y_norm,
            line_b_y_norm=line_b_y_norm,
            line_a_x_norm=line_a_x_norm,
            line_b_x_norm=line_b_x_norm,
        )

        # 4. Transcode raw intermediate video to universal HTML5-compatible H.264 via FFmpeg
        # -preset ultrafast minimizes transcoding latency down to ~1-2 seconds with -y overwrite
        ffmpeg_bin = get_ffmpeg_binary()
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            raw_output_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            final_h264_path,
        ]

        logger.info("Transcoding via FFmpeg: %s", " ".join(ffmpeg_cmd))
        try:
            ffmpeg_res = await asyncio.to_thread(
                subprocess.run,
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if ffmpeg_res.returncode != 0:
                logger.warning("FFmpeg returncode non-zero (%d): %s. Attempting fallback copy.", ffmpeg_res.returncode, ffmpeg_res.stderr)
        except Exception as ffmpeg_err:
            logger.warning("FFmpeg transcoding exception: %s. Using raw output fallback.", ffmpeg_err)

        # Fallback guarantee: if H.264 video was not created or empty, copy raw output directly
        if not os.path.exists(final_h264_path) or os.path.getsize(final_h264_path) == 0:
            if os.path.exists(raw_output_path):
                shutil.copy2(raw_output_path, final_h264_path)
                logger.info("Copied raw output video to final destination '%s'.", final_h264_path)

        # 5. Delete intermediate raw video and uploaded raw input video to keep disk clean
        for temp_file in (temp_input_path, raw_output_path):
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug("Deleted temporary file: %s", temp_file)
                except OSError as err:
                    logger.warning("Failed to remove temporary file '%s': %s", temp_file, err)

        # 6. Return structured Pydantic response guaranteed to contain metrics, performance, and events
        return ProcessVideoResponse(
            status="success",
            video_filename=final_h264_filename,
            metrics=MetricsResponse(
                total_in=int(result.get("total_in", 0)),
                total_out=int(result.get("total_out", 0)),
                current_occupancy=int(result.get("occupancy", 0)),
                total_tracks=int(result.get("total_tracks", 0)),
            ),
            performance=PerformanceResponse(
                avg_fps=float(result.get("avg_fps", 0.0)),
                total_frames=int(result.get("total_frames", 0)),
            ),
            events=[EventItem(**evt) for evt in result.get("events", [])],
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions directly
        raise
    except Exception as exc:
        logger.error("Error processing video upload: %s", exc, exc_info=True)
        # Clean up any partial files on error
        for cleanup_path in (temp_input_path, raw_output_path, final_h264_path):
            if os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                except OSError:
                    pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(exc)}",
        )


@app.get(
    "/videos/{filename}",
    summary="Stream or download processed video",
    response_class=FileResponse,
)
async def get_video(filename: str) -> FileResponse:
    """Serve requested processed MP4 video file with HTML5 media_type='video/mp4'."""
    # Prevent path traversal attacks
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(RUNS_DIR, safe_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file '{safe_filename}' not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=safe_filename,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
