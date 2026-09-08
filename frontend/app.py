"""
app.py - Streamlit Frontend for ProGlint Footfall Analytics (RT-DETR + BoT-SORT).
"""

import os
import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="ProGlint Footfall Hub", page_icon="🚶", layout="wide")


def extract_first_frame(video_bytes: bytes, ext: str = ".mp4") -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    ret, frame = cap.read()
    cap.release()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return frame if ret and frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)


def draw_calibration_lines(frame: np.ndarray, line_a_norm: float = 0.45, line_b_norm: float = 0.55) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame.copy()
    h, w = rgb.shape[:2]
    ya, yb = int(line_a_norm * h), int(line_b_norm * h)
    cv2.line(rgb, (0, ya), (w, ya), (0, 255, 255), 2)  # Line A (Cyan/Yellow)
    cv2.putText(rgb, "Line A (In)", (10, max(20, ya - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(rgb, (0, yb), (w, yb), (255, 0, 255), 2)  # Line B (Magenta)
    cv2.putText(rgb, "Line B (Out)", (10, max(20, yb - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return rgb


def build_occupancy_chart(events: list, total_duration_sec: float = 10.0) -> go.Figure:
    times, occ = [0.0], [0]
    curr = 0
    for ev in sorted(events, key=lambda x: x.get("time_sec", 0)):
        t = ev.get("time_sec", 0)
        curr += 1 if ev.get("type") == "IN" else -1
        times.extend([t, t])
        occ.extend([occ[-1], curr])
    times.append(max(total_duration_sec, times[-1]))
    occ.append(curr)

    fig = go.Figure(data=[go.Scatter(x=times, y=occ, mode="lines", line_shape="hv", line=dict(color="#00ffcc", width=2))])
    fig.update_layout(title="Net Occupancy Over Time", xaxis_title="Time (s)", yaxis_title="Occupancy", template="plotly_dark", height=280)
    return fig


# --- Streamlit UI Layout ---
st.title("🚶 ProGlint Footfall Hub (RT-DETR + BoT-SORT)")
api_url = st.sidebar.text_input("Backend API URL", value="http://127.0.0.1:8000")

# Check backend health
try:
    h_resp = requests.get(f"{api_url}/health", timeout=2).json()
    st.sidebar.success(f"Backend Connected: {h_resp.get('hardware', 'CPU')}")
except Exception:
    st.sidebar.error("Backend Offline. Run: uvicorn backend.main:app")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload Surveillance Video", type=["mp4", "avi", "mov"])
    line_a = st.slider("Line A (Entry) Position", 0.0, 1.0, 0.45, 0.05)
    line_b = st.slider("Line B (Exit) Position", 0.0, 1.0, 0.55, 0.05)
    process_btn = st.button("🚀 Process Video", type="primary", disabled=not uploaded)

if uploaded:
    video_bytes = uploaded.read()
    with col2:
        first_frame = extract_first_frame(video_bytes)
        st.image(draw_calibration_lines(first_frame, line_a, line_b), caption="Tripwire Calibration Preview", use_container_width=True)

    if process_btn:
        with st.spinner("Processing video with RT-DETR + BoT-SORT on GPU..."):
            files = {"video": (uploaded.name, video_bytes, uploaded.type or "video/mp4")}
            data = {"line_a_norm": str(line_a), "line_b_norm": str(line_b)}
            res = requests.post(f"{api_url}/process_video", files=files, data=data).json()

        m = res.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total IN", m.get("total_in", 0))
        c2.metric("Total OUT", m.get("total_out", 0))
        c3.metric("Net Occupancy", m.get("current_occupancy", 0))
        c4.metric("Avg FPS", res.get("performance", {}).get("avg_fps", 0))

        vid_name = res.get("video_filename")
        if vid_name:
            vid_res = requests.get(f"{api_url}/videos/{vid_name}")
            st.video(vid_res.content)
            st.plotly_chart(build_occupancy_chart(res.get("events", [])), use_container_width=True)
