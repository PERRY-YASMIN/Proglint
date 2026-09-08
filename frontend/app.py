"""
Footfall Analytics Hub - Streamlit Frontend Application.
Interfaces with the FastAPI Footfall Tracking Service.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Footfall Analytics Hub | Proglint Evaluation",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, high-contrast dashboard aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E88E5;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #757575;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3440;
    }
    .tripwire-legend {
        display: flex;
        gap: 20px;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    .legend-a {
        color: #00E5FF;
        font-weight: 600;
    }
    .legend-b {
        color: #FF00FF;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def query_health(api_url: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Query FastAPI /health endpoint and return status, response data, and error message."""
    try:
        url = f"{api_url.rstrip('/')}/health"
        resp = requests.get(url, timeout=3.0)
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"HTTP Error {resp.status_code}"
    except requests.exceptions.RequestException as exc:
        return False, None, str(exc)


def extract_first_frame(video_bytes: bytes, file_ext: str) -> Optional[np.ndarray]:
    """Extract the first frame from uploaded video bytes using OpenCV without disk leakage."""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
            f.write(video_bytes)
            temp_file = f.name

        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            return None

        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            return frame
        return None
    except Exception as exc:
        st.error(f"Error reading initial video frame: {exc}")
        return None
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


def draw_calibration_lines(
    frame: np.ndarray,
    line_a_norm: float,
    line_b_norm: float,
    orientation: str = "horizontal",
) -> np.ndarray:
    """Draw virtual tripwires on frame copy for calibration preview."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    clean_orientation = orientation.lower().strip() if orientation else "horizontal"
    if clean_orientation == "vertical":
        x_a = int(line_a_norm * w)
        x_b = int(line_b_norm * w)

        # Line A: Cyan (BGR: 255, 255, 0)
        cv2.line(annotated, (x_a, 0), (x_a, h), (255, 255, 0), 2, cv2.LINE_AA)
        label_x_a = int(max(10, min(w - 220, x_a + 8)))
        cv2.putText(
            annotated,
            f"Line A (Outer): {line_a_norm:.2f}",
            (label_x_a, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Line B: Magenta (BGR: 255, 0, 255)
        cv2.line(annotated, (x_b, 0), (x_b, h), (255, 0, 255), 2, cv2.LINE_AA)
        label_x_b = int(max(10, min(w - 220, x_b + 8)))
        cv2.putText(
            annotated,
            f"Line B (Inner): {line_b_norm:.2f}",
            (label_x_b, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        y_a = int(line_a_norm * h)
        y_b = int(line_b_norm * h)

        # Line A: Cyan (BGR: 255, 255, 0)
        cv2.line(annotated, (0, y_a), (w, y_a), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"Line A (Outer): {line_a_norm:.2f}",
            (15, max(24, y_a - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Line B: Magenta (BGR: 255, 0, 255)
        cv2.line(annotated, (0, y_b), (w, y_b), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"Line B (Inner): {line_b_norm:.2f}",
            (15, max(24, y_b - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Convert BGR to RGB for Streamlit rendering
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def build_occupancy_chart(
    events: List[Dict[str, Any]], total_duration_sec: float = 0.0
) -> go.Figure:
    """Construct an analytical step-chart of Net Occupancy over the video timeline."""
    fig = go.Figure()

    if not events:
        fig.add_annotation(
            text="No crossing events recorded in this session.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#9E9E9E"),
        )
        fig.update_layout(
            title="Net Occupancy Over Time",
            xaxis_title="Timeline (seconds)",
            yaxis_title="Occupancy Count",
            template="plotly_dark",
            height=340,
        )
        return fig

    # Chronologically sort events
    sorted_events = sorted(events, key=lambda x: x["time_sec"])

    timeline_points = [0.0]
    occupancy_points = [0]
    running_net = 0

    for evt in sorted_events:
        t = evt["time_sec"]
        if evt["type"] == "IN":
            running_net += 1
        elif evt["type"] == "OUT":
            running_net -= 1

        timeline_points.append(t)
        occupancy_points.append(running_net)

    # Extend to end of video if duration provided
    max_t = max(timeline_points[-1], total_duration_sec)
    if max_t > timeline_points[-1]:
        timeline_points.append(max_t)
        occupancy_points.append(running_net)

    fig.add_trace(
        go.Scatter(
            x=timeline_points,
            y=occupancy_points,
            mode="lines+markers",
            line_shape="hv",  # Step-wise discrete transitions
            line=dict(color="#00E5FF", width=2.5),
            marker=dict(size=6, color="#FF00FF"),
            name="Net Occupancy",
            hovertemplate="Time: %{x:.2f}s<br>Net Occupancy: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Real-Time Occupancy Flow Over Timeline",
        xaxis_title="Video Timestamp (seconds)",
        yaxis_title="Net Occupancy",
        template="plotly_dark",
        height=350,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# --- Application Layout ---

st.markdown('<div class="main-title">🚶 Footfall Analytics Hub</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production Computer Vision Pipeline powered by YOLO11, ByteTrack, and Dual Virtual Tripwires</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.header("⚙️ System & Calibration Settings")

api_base_url = st.sidebar.text_input(
    "Backend API Base URL",
    value="http://127.0.0.1:8000",
    help="FastAPI backend host address",
)

# Backend Health Check
is_healthy, health_data, err_msg = query_health(api_base_url)
if is_healthy and health_data:
    device_name = health_data.get("device", "Unknown")
    cuda_status = "CUDA Active" if health_data.get("cuda_available") else "CPU Mode"
    st.sidebar.success(f"🟢 Backend Online\n• Device: `{device_name}`\n• Mode: `{cuda_status}`")
else:
    st.sidebar.error(
        f"🔴 Backend Offline\n{err_msg or 'Cannot connect'}\n\n*Ensure `backend/main.py` is running on {api_base_url}*"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Tripwire Plane Calibration")

orientation_choice = st.sidebar.radio(
    "Gate Orientation",
    ["Horizontal (Top/Bottom Flow)", "Vertical (Left/Right Flow)"],
    help="Select tripwire orientation based on pedestrian walking direction in the video.",
)
orientation_val = "vertical" if "Vertical" in orientation_choice else "horizontal"

axis_name = "horizontal" if orientation_val == "vertical" else "vertical"
pos_a_desc = "0.0=Left, 1.0=Right" if orientation_val == "vertical" else "0.0=Top, 1.0=Bottom"
pos_b_desc = "0.0=Left, 1.0=Right" if orientation_val == "vertical" else "0.0=Top, 1.0=Bottom"

line_a_norm = st.sidebar.slider(
    "Line A (Outer - Cyan)",
    min_value=0.05,
    max_value=0.95,
    value=0.45,
    step=0.01,
    help=f"Normalized {axis_name} position ({pos_a_desc}). Pedestrians cross Line A first when entering.",
)

line_b_norm = st.sidebar.slider(
    "Line B (Inner - Magenta)",
    min_value=0.05,
    max_value=0.95,
    value=0.55,
    step=0.01,
    help=f"Normalized {axis_name} position ({pos_b_desc}). Pedestrians cross Line B to complete entry.",
)

timeout_sec = st.sidebar.slider(
    "FSM Crossing Timeout (s)",
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
    help="Max duration allowed between crossing first and second tripwire before state resets to IDLE.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance Optimization")

frame_stride = st.sidebar.slider(
    "Inference Frame Stride",
    min_value=1,
    max_value=6,
    value=1,
    step=1,
    help="Stride for YOLO11 inference (1 = full per-frame processing, optimal on RTX 4060 GPU).",
)

# Main Section
upload_col, preview_col = st.columns([1, 1], gap="medium")

with upload_col:
    st.subheader("📹 Video Source Ingestion")
    uploaded_file = st.file_uploader(
        "Upload pedestrian surveillance video",
        type=["mp4", "avi", "mov"],
        help="Supports standard formats (.mp4, .avi, .mov)",
    )

first_frame: Optional[np.ndarray] = None
preview_drawn: Optional[np.ndarray] = None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"

    first_frame = extract_first_frame(file_bytes, file_ext)

    with preview_col:
        st.subheader("🎯 Dual Tripwire Calibration Preview")
        if first_frame is not None:
            preview_drawn = draw_calibration_lines(
                first_frame, line_a_norm, line_b_norm, orientation=orientation_val
            )
            st.image(
                preview_drawn,
                caption=f"Tripwire Overlay Preview ({first_frame.shape[1]}x{first_frame.shape[0]}px - {orientation_val.capitalize()})",
                width="stretch",
            )
            st.markdown(
                '<div class="tripwire-legend">'
                '<span class="legend-a">━ Line A: Outer Boundary</span>'
                '<span class="legend-b">━ Line B: Inner Boundary</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Unable to render first frame preview. Video format may require transcoding.")

    # Processing Trigger Section
    st.markdown("---")
    btn_col, status_col = st.columns([1, 2])

    with btn_col:
        start_processing = st.button(
            "🚀 Start Footfall Analysis",
            type="primary",
            width="stretch",
            disabled=not is_healthy,
        )

    if start_processing:
        if not is_healthy:
            st.error("Cannot process video: FastAPI backend is offline.")
        else:
            with st.spinner("⏳ Analyzing pedestrian trajectories with YOLO11 & ByteTrack..."):
                try:
                    process_url = f"{api_base_url.rstrip('/')}/process_video"
                    files = {
                        "video": (uploaded_file.name, file_bytes, uploaded_file.type or "video/mp4")
                    }
                    data = {
                        "line_a_norm": str(line_a_norm),
                        "line_b_norm": str(line_b_norm),
                        "timeout_sec": str(timeout_sec),
                        "frame_stride": str(frame_stride),
                        "orientation": orientation_val,
                    }

                    response = requests.post(process_url, files=files, data=data, timeout=None)

                    if response.status_code == 200:
                        st.session_state["result_payload"] = response.json()
                        st.success("✅ Analysis completed and video transcoded successfully!")
                    else:
                        st.error(
                            f"Backend error ({response.status_code}): {response.text}"
                        )
                except requests.exceptions.RequestException as req_err:
                    st.error(f"Network error while communicating with backend: {req_err}")

# Render Results Section if available in session_state
if "result_payload" in st.session_state:
    res = st.session_state["result_payload"]
    metrics = res.get("metrics", {})
    perf = res.get("performance", {})
    events = res.get("events", [])
    video_filename = res.get("video_filename")

    st.markdown("---")
    st.subheader("📊 Footfall Performance & Operational Metrics")

    # 4 Metric Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    delta_in = (
        "Line A → Line B (Left → Right)"
        if orientation_val == "vertical"
        else "Line A → Line B (Top → Bottom)"
    )
    delta_out = (
        "- Line B → Line A (Right → Left)"
        if orientation_val == "vertical"
        else "- Line B → Line A (Bottom → Top)"
    )

    with kpi_col1:
        st.metric(
            label="Total Entries (IN)",
            value=metrics.get("total_in", 0),
            delta=delta_in,
        )

    with kpi_col2:
        st.metric(
            label="Total Exits (OUT)",
            value=metrics.get("total_out", 0),
            delta=delta_out,
            delta_color="inverse",
        )

    with kpi_col3:
        current_occ = metrics.get("current_occupancy", 0)
        st.metric(
            label="Current Occupancy (Net)",
            value=current_occ,
            delta=f"Unique Tracks: {metrics.get('total_tracks', 0)}",
        )

    with kpi_col4:
        avg_fps = perf.get("avg_fps", 0.0)
        total_frames = perf.get("total_frames", 0)
        st.metric(
            label="Processing Throughput",
            value=f"{avg_fps:.1f} FPS",
            delta=f"{total_frames} Frames",
        )

    # Video & Analytics Columns
    st.markdown("### 🎬 Annotated Video Playback & Flow Analysis")
    vid_display_col, chart_col = st.columns([1.1, 0.9], gap="large")

    with vid_display_col:
        st.markdown("**Rendered H.264 Stream (HUD + Tripwires + Bounding Boxes)**")
        video_url = f"{api_base_url.rstrip('/')}/videos/{video_filename}"

        try:
            # Stream directly or fetch bytes
            vid_resp = requests.get(video_url, timeout=30)
            if vid_resp.status_code == 200:
                st.video(vid_resp.content, format="video/mp4")
                st.download_button(
                    label="💾 Download Processed Video (.mp4)",
                    data=vid_resp.content,
                    file_name=video_filename,
                    mime="video/mp4",
                    width="stretch",
                )
            else:
                st.error(f"Could not load processed video: HTTP {vid_resp.status_code}")
        except Exception as vid_err:
            st.error(f"Error fetching video stream: {vid_err}")

    with chart_col:
        approx_duration = (
            float(perf.get("total_frames", 0)) / 30.0 if perf.get("avg_fps", 0) > 0 else 0.0
        )
        occ_fig = build_occupancy_chart(events, approx_duration)
        st.plotly_chart(occ_fig, width="stretch")

    # Audit Trail Table
    st.markdown("### 📋 Event Verification & Audit Trail")
    with st.expander("Detailed Crossing Event Audit Log", expanded=True):
        if events:
            df_events = pd.DataFrame(events)
            df_events.rename(
                columns={
                    "frame": "Frame Index",
                    "time_sec": "Timestamp (s)",
                    "id": "Track ID",
                    "type": "Direction Event",
                },
                inplace=True,
            )
            st.dataframe(
                df_events,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No pedestrian crossings registered by the tripwire state machine.")

else:
    if uploaded_file is None:
        st.info("👆 Please upload a video file above to calibrate virtual tripwires and initiate tracking.")

# Memory Cleanup
del first_frame
del preview_drawn
