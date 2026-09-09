"""frontend/app.py - Minimal Streamlit UI for ProGlint Footfall Intelligence."""
import os, tempfile, cv2, requests, numpy as np, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="ProGlint Footfall Hub", layout="wide")

def extract_first_frame(video_bytes: bytes, ext: str = ".mp4") -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(video_bytes)
        p = tmp.name
    cap = cv2.VideoCapture(p)
    ret, frame = cap.read()
    cap.release()
    if os.path.exists(p): os.remove(p)
    return frame if ret and frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)

def draw_calibration_lines(frame: np.ndarray, line_a: float = 0.45, line_b: float = 0.55, **kwargs) -> np.ndarray:
    line_a = kwargs.get("line_a_norm", line_a)
    line_b = kwargs.get("line_b_norm", line_b)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame.copy()
    h, w = rgb.shape[:2]
    ya, yb = int(line_a * h), int(line_b * h)
    cv2.line(rgb, (0, ya), (w, ya), (0, 255, 255), 2)
    cv2.putText(rgb, "Line A (Outer)", (10, max(20, ya - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(rgb, (0, yb), (w, yb), (255, 0, 255), 2)
    cv2.putText(rgb, "Line B (Inner)", (10, max(20, yb - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return rgb

def build_occupancy_chart(events: list, total_duration_sec: float = 10.0, **kwargs) -> go.Figure:
    times, occ, curr = [0.0], [0], 0
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

# Sidebar Calibration Controls
st.sidebar.title("Tripwire Calibration")
api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")
line_a = st.sidebar.slider("Line A (Cyan) Position", 0.05, 0.95, 0.45, 0.01)
line_b = st.sidebar.slider("Line B (Magenta) Position", 0.05, 0.95, 0.55, 0.01)
timeout_s = st.sidebar.slider("FSM Crossing Timeout (s)", 1.0, 10.0, 3.0, 0.5)

st.title("ProGlint Footfall Intelligence Hub")
uploaded = st.file_uploader("Upload Surveillance Video", type=["mp4", "avi", "mov"])

if uploaded:
    video_bytes = uploaded.read()
    col_prev, col_act = st.columns([1, 1])
    with col_prev:
        st.subheader("Tripwire Alignment Preview")
        preview = draw_calibration_lines(extract_first_frame(video_bytes), line_a, line_b)
        st.image(preview, use_container_width=True)

    with col_act:
        st.subheader("Inference Execution")
        if st.button("Start Footfall Analysis", type="primary"):
            with st.spinner("Processing video..."):
                files = {"video": (uploaded.name, video_bytes, uploaded.type or "video/mp4")}
                data = {"line_a_norm": str(line_a), "line_b_norm": str(line_b), "timeout_sec": str(timeout_s)}
                try:
                    res = requests.post(f"{api_url}/process_video", files=files, data=data, timeout=None).json()
                    st.session_state["res"] = res
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    if "res" in st.session_state:
        res = st.session_state["res"]
        m, p = res.get("metrics", {}), res.get("performance", {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Entries (IN)", m.get("total_in", 0))
        k2.metric("Total Exits (OUT)", m.get("total_out", 0))
        k3.metric("Current Occupancy", m.get("current_occupancy", 0))
        k4.metric("Throughput (FPS)", p.get("avg_fps", 0.0))

        v_col, g_col = st.columns([1, 1])
        with v_col:
            st.subheader("Processed Video Output")
            vid_res = requests.get(f"{api_url}/videos/{res.get('video_filename')}")
            st.video(vid_res.content)
        with g_col:
            st.subheader("Occupancy Timeline")
            st.plotly_chart(build_occupancy_chart(res.get("events", [])), use_container_width=True)

        events = res.get("events", [])
        if events:
            with st.expander("Crossing Event Log"):
                st.dataframe(events, use_container_width=True)