# ProGlint Footfall Intelligence — Code Pinpoint & Quick Reference Guide

> **Purpose of this guide:** If your professor, evaluator, or sir asks you:
> - *"Which is the main file I need to see?"*
> - *"Where is the model initialized?"*
> - *"Where is the code coming from and where is it used?"*
> - *"How does the directional line crossing logic function?"*
> - *"Why are files organized this way?"*
>
> This document gives you the exact answer, directory layout, code comment tags, and file line pointers.

---

## 1. Quick Answer: Which File Do I Show Sir?

| What Sir Wants to See | The Exact File to Open | Why This File |
| :--- | :--- | :--- |
| **⭐ The Core AI & Logic (MAIN FILE)** | [`engine/tracker.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/engine/tracker.py) | **Show this first!** It contains the entire computer vision intelligence: RT-DETR model initialization, BoT-SORT tracking, foot anchor point calculation, 2-line directional FSM, and HUD visualization. |
| **The Full System Running** | [`run.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/run.py) | Run `python run.py both` to show FastAPI backend + Streamlit frontend launching together. |
| **The Live UI Dashboard** | [`frontend/app.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/frontend/app.py) | Streamlit dashboard with interactive tripwire calibration sliders, video player, KPI counters, and occupancy timeline. |
| **The REST API Backend** | [`backend/main.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/backend/main.py) | FastAPI service with `/process_video`, `/health`, and `/videos/{filename}` endpoints. |
| **Model Fine-Tuning** | [`data/scripts/train_rtdetr.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/data/scripts/train_rtdetr.py) | Script that fine-tunes RT-DETR on PETS-2009 access-control dataset on CUDA device 0. |
| **PETS-2009 Ingestion** | [`data/scripts/convert_pets2009.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/data/scripts/convert_pets2009.py) | Parses CVML XML, clamps coordinates, splits train/val/test, and creates `pets2009.yaml`. |
| **Verification & Tests** | [`tests/`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/tests) | 14 unit and integration tests verifying engine math, API contracts, dataset conversions, and E2E pipeline. |

---

## 2. Standardized Code Annotation Tags

Every line of logic across all Python files is annotated with 5 standard tags:

1. **`[FROM: ...]`** — Shows where an import, library, or dependency comes from (e.g., Python Standard Library, PyTorch, Ultralytics, OpenCV, or internal modules).
2. **`[DEF: ...]`** — Shows what function, class, variable, or endpoint is being defined.
3. **`[INIT: ...]`** — Shows where models, devices (CUDA/CPU), state trackers, or data structures are initialized.
4. **`[FUNCTION: ...]`** — Explains the underlying mechanics, math, algorithm, or state transitions (e.g., Euclidean distance, FSM transitions, coordinate normalization).
5. **`[USED IN: ...]`** — Shows which downstream function, endpoint, UI component, or caller uses this logic.

---

## 3. Clean Project Directory Layout

```
PROGLINT PROXY/
│
├── engine/                      # Core Computer Vision Engine
│   ├── __init__.py
│   └── tracker.py               # ⭐ MAIN FILE: FootfallEngine (RT-DETR + BoT-SORT + FSM)
│
├── backend/                     # REST API Microservice
│   ├── __init__.py
│   └── main.py                  # FastAPI server (/process_video, /health, /videos/{filename})
│
├── frontend/                    # Web Dashboard
│   └── app.py                   # Streamlit UI with sliders, Plotly chart, and video player
│
├── data/                        # Data Management
│   ├── processed/               # Normalized YOLO images, labels, pets2009.yaml
│   ├── raw/                     # Original PETS-2009 CVML XML annotations & frames
│   └── scripts/                 # Training & dataset conversion scripts
│       ├── convert_pets2009.py  # Ingestion & conversion of PETS-2009 benchmark
│       ├── train_rtdetr.py      # Fine-tunes RT-DETR-L on CUDA device 0
│       └── ...                  # Standalone data inspection/validation helpers
│
├── evaluation/                  # Latency & Accuracy Benchmarking
│   ├── __init__.py
│   ├── benchmark.py             # Latency & sustained FPS profiler on test video
│   └── BENCHMARK_REPORT.md      # Auto-generated benchmark report
│
├── tests/                       # Automated Test Suite (100% Passing)
│   ├── __init__.py
│   ├── test_footfall_engine.py  # Engine math, FSM crossing, and video processing tests
│   ├── test_backend_api.py      # FastAPI endpoint contract tests
│   ├── test_e2e.py              # End-to-end integration pipeline tests
│   └── test_convert_pets2009.py # PETS-2009 XML ingestion & normalization tests
│
├── runs/                        # Video & Training Artifacts
│   ├── custom_train/best_rtdetr.pt # Fine-tuned model checkpoint
│   └── ...                      # Temporary processed videos & logs
│
├── run.py                       # Single-command orchestrator (run.py backend / frontend / both)
├── rtdetr-l.pt                  # Pre-trained base RT-DETR model weights (66 MB)
├── TownCentre_test.mp4          # Verification & demo CCTV video clip
└── requirements.txt             # Python dependencies
```

---

## 4. Master "What Is Where & What Is Doing What" Index

### A. Core Engine & Inference (`engine/tracker.py` — ~250 lines with full comments)

| Feature / Logic | Where in `engine/tracker.py` | What It Does & Where Used |
| :--- | :--- | :--- |
| **Model Weights Path** | `CUSTOM_WEIGHTS_PATH`, `DEFAULT_MODEL` | Points to fine-tuned `runs/custom_train/best_rtdetr.pt` (fallback `rtdetr-l.pt`). Used in `__init__`. |
| **Path Resolver** | `get_default_model_path()` | Returns preferred model path; allows clean test mocking. |
| **Model Initialization** | `FootfallEngine.__init__()` | `[INIT: Model Load]` loads `RTDETR(model_path)`. `[INIT: Device]` selects CUDA GPU 0 if available, else CPU. |
| **State Reset** | `FootfallEngine.reset()` | Resets counters (`total_in=0`, `total_out=0`), seen IDs set, and tracking history. |
| **Net Occupancy** | `@property def occupancy` | `[FUNCTION: Occupancy]` Computes `total_in - total_out`. Used in HUD & API response. |
| **Detection & Tracking** | `self.model.track(...)` | `[FUNCTION: RT-DETR + BoT-SORT]` Runs inference for person class (`classes=[0]`) and Kalman tracking with Re-ID. |
| **Foot Anchor Point** | `cx = (x1 + x2) // 2`, `cy = y2` | Uses bottom-center of bounding box $(c_x, y_2)$ for ground plane contact. |
| **Displacement Filter** | `abs(cy - start_y) > 15` | Filters out stationary pedestrians shifting weight inside the tripwire zone. |
| **Dual-Tripwire FSM** | State Machine Logic | **Downward (Line A $\rightarrow$ Line B)** = `IN`<br>**Upward (Line B $\rightarrow$ Line A)** = `OUT`<br>Has timeout (default 45 frames) to expire incomplete crossings. |
| **Memory Cleanup** | Active Tracks Purge | Drops tracks absent for $>60$ frames to prevent memory leaks during long runs. |
| **HUD Visualization** | `cv2.line`, `cv2.putText` | Draws Cyan Line A, Magenta Line B, bounding boxes, track IDs, and real-time counter bar. |
| **Video Processing Loop** | `FootfallEngine.process_video()` | Reads video via OpenCV VideoCapture, processes frame-by-frame, writes output MP4. |

---

### B. Backend Microservice (`backend/main.py` — ~140 lines with full comments)

| Feature / Logic | Endpoint / Function | What It Does & Where Used |
| :--- | :--- | :--- |
| **RUNS_DIR** | `RUNS_DIR = os.path.abspath("runs")` | Temporary storage for uploaded and processed videos. |
| **FastAPI Setup** | `app = FastAPI(...)` + `CORSMiddleware` | Enables cross-origin communication with Streamlit on port 8501. |
| **Global Engine** | `engine = FootfallEngine()` | Initializes singleton model in RAM on backend startup to avoid reloading per request. |
| **H.264 Transcoding** | `to_h264(src, dst)` | Invokes FFmpeg (`libx264`, `yuv420p`) so MP4 videos can stream in HTML5 browsers. |
| **Health Check** | `GET /health` | Returns server status, device (`cuda:0` / `cpu`), detector name, and tracker name. |
| **Video Ingestion** | `POST /process_video` | Receives video upload, Line A/B coordinates, executes `engine.process_video()`, transcodes, and returns JSON metrics. |
| **Video Streaming** | `GET /videos/{filename}` | Streams processed video file with `video/mp4` MIME type to the frontend. |

---

### C. Frontend Dashboard (`frontend/app.py` — ~140 lines with full comments)

| Feature / Logic | Function / UI Block | What It Does & Where Used |
| :--- | :--- | :--- |
| **First Frame Extraction** | `extract_first_frame(video_bytes)` | Reads frame 0 via OpenCV from uploaded bytes to enable real-time line calibration. |
| **Calibration Preview** | `draw_calibration_lines(frame, y_a, y_b)` | Renders Cyan Line A and Magenta Line B over frame 0 matching sidebar slider positions. |
| **Occupancy Timeline** | `build_occupancy_chart(event_log)` | Builds interactive Plotly step-chart showing net occupancy changes across time. |
| **Sidebar Calibration** | `st.sidebar.slider` | Lets operator adjust Line A, Line B, and timeout frames interactively. |
| **KPI Metrics Display** | `st.metric` | Displays Total Entries (IN), Total Exits (OUT), Net Occupancy, and Sustained FPS. |
| **Video Playback** | `st.video` | Streams output video from `backend:8000/videos/{filename}`. |
| **Audit Event Table** | `st.dataframe` | Displays chronological event table with timestamps, track IDs, and crossing directions. |

---

### D. Training & Ingestion Pipelines

| File Path | Main Function | What It Does |
| :--- | :--- | :--- |
| [`data/scripts/train_rtdetr.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/data/scripts/train_rtdetr.py) | `model.train(...)` | Fine-tunes `rtdetr-l.pt` on `data/processed/pets2009.yaml` for 15 epochs on CUDA device 0. Copies checkpoint to `runs/custom_train/best_rtdetr.pt`. |
| [`data/scripts/convert_pets2009.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/data/scripts/convert_pets2009.py) | `prepare_pets2009_dataset()`, `verify_dataset()` | Ingests PETS-2009 CVML XML annotations, normalizes coordinates $[0, 1]$, splits train/val/test, and creates `pets2009.yaml`. |
| [`evaluation/benchmark.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/evaluation/benchmark.py) | `run_benchmark()` | Runs inference on `TownCentre_test.mp4`, profiles latency in ms/frame and sustained FPS, generates `BENCHMARK_REPORT.md`. |
| [`run.py`](file:///C:/Users/MOHAMMAD%20SAJID/OneDrive/Desktop/PROGLINT%20PROXY/run.py) | `run_backend()`, `run_frontend()` | Single-command launcher for backend (`run.py backend`), frontend (`run.py frontend`), or both (`run.py both`). |

---

## 5. How to Run Everything

```powershell
# 1. Run all 14 Unit and Integration Tests (from root)
.\cv_env\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"

# 2. Launch the Full Application (Backend + Frontend together)
.\cv_env\Scripts\python.exe run.py both

# 3. Launch Backend only (FastAPI on http://127.0.0.1:8000)
.\cv_env\Scripts\python.exe run.py backend

# 4. Launch Frontend only (Streamlit on http://localhost:8501)
.\cv_env\Scripts\python.exe run.py frontend

# 5. Run Performance & Latency Benchmark
.\cv_env\Scripts\python.exe evaluation/benchmark.py --video TownCentre_test.mp4 --max_frames 140

# 6. Fine-Tune RT-DETR on PETS-2009 Dataset
.\cv_env\Scripts\python.exe data/scripts/train_rtdetr.py
```
