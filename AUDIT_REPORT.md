# ProGlint Footfall Intelligence: Project Audit, Workspace Clean-Up & Baseline Refactoring Report

**Author:** Senior MLOps & Computer Vision Architect  
**Project:** ProGlint Footfall Intelligence  
**Objective:** Transition from YOLO11n + ByteTrack baseline to domain-adapted RT-DETR (RT-DETR-R18) + BoT-SORT on PETS-2009 surveillance benchmark.  
**Audit Timestamp:** 2026-09-08T15:10:00+05:30  

---

## 1. Executive Summary

A comprehensive architectural audit, workspace cleanup, dependency alignment, and modular refactoring was performed on the ProGlint Footfall Analytics platform. The system has been cleanly restructured to decouple detection architectures (`engine/detector.py`), tracking algorithms (`engine/tracker.py`), directional tripwire state machines (`engine/fsm_counter.py`), benchmark evaluation suites (`evaluation/benchmark.py`), and dataset preparation pipelines (`data/scripts/convert_pets2009.py`).

Crucially, **100% backward compatibility** has been maintained: all 31 unit, API, and end-to-end integration tests pass without regression.

---

## 2. Process & Port Cleanup

Prior to the audit, lingering background Python microservices were identified binding to active network ports. These processes were safely terminated to release system resources and free default ports:

| Process ID | Name | Command Line | Port | Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| **17120** | `python.exe` | `python run.py` (Supervising launcher) | N/A | **Terminated** |
| **14992** | `python3.12.exe` | `-m uvicorn backend.main:app --port 8000` | `8000` (FastAPI) | **Terminated** |
| **11592** | `python3.12.exe` | `-m streamlit run frontend/app.py --server.port 8501` | `8501` (Dashboard) | **Terminated** |

### Port Verification Status
- **Port 8000** (FastAPI Microservice): `FREE` (No active TCP listeners)
- **Port 8501** (Streamlit Dashboard): `FREE` (No active TCP listeners)

---

## 3. Structural Housekeeping & File Retention Audit

A total of **~2.04 GB** of orphaned video transcode fragments and temporary evaluation predictions were cleaned up from disk, while valid fine-tuned weights, checkpoints, logs, and benchmark videos were strictly preserved.

### Cleaned Stale Files (Removed)

| File / Pattern | Quantity | Disk Space Recovered | Description / Rationale |
| :--- | :---: | :---: | :--- |
| `runs/*.mp4` | 20 files | **2,044,813,514 bytes (~2.04 GB)** | Stale transcode outputs and temporary test stream fragments (`03d74ee5eebb_processed.mp4`, `temp_input_*.mp4`, `raw_output_*.mp4`, etc.) |
| `runs/custom_train/exp/val_batch0_pred.jpg` | 1 file | **212,242 bytes (~207 KB)** | Stale inference prediction test artifact from prior training run |

### Retained Critical Assets (Preserved)

| File / Directory Path | Size | Status | Purpose & Downstream Role |
| :--- | :---: | :---: | :--- |
| `yolo11n.pt` | 5.61 MB | **Preserved** | Official Ultralytics base weights for comparative baseline benchmarking |
| `runs/custom_train/best.pt` | 5.44 MB | **Preserved** | Valid custom fine-tuned pedestrian model weights |
| `runs/custom_train/exp/weights/best.pt` | 5.44 MB | **Preserved** | Checkpoint backup |
| `runs/custom_train/exp/weights/last.pt` | 5.44 MB | **Preserved** | Last epoch training checkpoint |
| `runs/custom_train/exp/args.yaml`, `results.csv`, `*.png` | ~800 KB | **Preserved** | Training hyperparameters, confusion matrices, and PR curves |
| `TownCentre_test.mp4` | ~15 MB | **Preserved** | Reference benchmark video for comparative pipeline evaluation |
| `data/pedestrian_mini/` & `data/pedestrian_mini.yaml` | ~48 images | **Preserved** | Functional mini-dataset for smoke testing and baseline regression |
| `test_footfall_engine.py` | 37.5 KB | **Preserved & Verified** | 25 Unit/integration tests for FSM crossing, bounding boxes, and frame strides |
| `test_backend_api.py` | 6.1 KB | **Preserved & Verified** | 5 API endpoint tests (/health, /process_video, /videos) |
| `test_e2e.py` | 4.0 KB | **Preserved & Verified** | Full E2E integration test across engine, API, and frontend helpers |

---

## 4. Git Version Control Audit (`.gitignore`)

The `.gitignore` manifest was audited and expanded to guarantee that datasets, model checkpoints, and runtime media artifacts are completely excluded from Git version control:

```gitignore
# Virtual Environments
cv_env/
venv/
env/
.venv/
ENV/

# Python Cache & Bytecode
__pycache__/
*.py[cod]
*$py.class
*.cache

# Datasets & Benchmarks
data/pets*
data/raw/
data/processed/
data/pedestrian_mini/
data/pedestrian_mini.yaml
*.zip
*.tar
*.tar.gz
*.tgz

# Model Checkpoints & Weights
*.pt
*.pth
*.onnx
*.engine
weights/

# Execution Runs & Media Outputs
runs/
*.mp4
*.avi
*.mov
*.mkv

# IDE & OS Files
.vscode/
.idea/
.DS_Store
Thumbs.db
*.swp
*.swo

# Temporary & Log Files
*.log
scratch/
```

---

## 5. Standardized Project Directory Scaffold

The codebase now conforms to the standardized architecture:

```
PROGLINT/
├── data/
│   ├── raw/                           # Incoming benchmark archives (PETS-2009 XMLs & images)
│   ├── processed/                     # Parsed YOLO-format annotations & YAML manifests
│   └── scripts/                       # Conversion & dataset preparation scripts
│       ├── __init__.py
│       └── convert_pets2009.py        # PETS-2009 XML to YOLO annotation converter & splitter
├── engine/
│   ├── __init__.py                    # Public engine API exports
│   ├── detector.py                    # Unified detector abstraction (RT-DETR & YOLO baseline)
│   ├── tracker.py                     # Multi-object tracker module (BoT-SORT & ByteTrack) + FootfallEngine
│   └── fsm_counter.py                 # Dual-tripwire directional Finite State Machine & TrackState
├── backend/
│   ├── __init__.py
│   └── main.py                        # FastAPI microservice
├── frontend/
│   └── app.py                         # Streamlit analytics dashboard
├── evaluation/
│   ├── __init__.py
│   └── benchmark.py                   # Comparative evaluation suite (mAP, HOTA, counting accuracy)
├── runs/                              # Dynamic artifacts (weights, logs, transcode caches)
│   └── custom_train/
│       ├── best.pt
│       └── exp/
├── cv_env/                            # Virtual environment
├── requirements.txt                   # Standardized dependencies
├── README.md                          # Project documentation
├── test_footfall_engine.py            # Unit test suite
├── test_backend_api.py                # Backend API test suite
├── test_e2e.py                        # System E2E test suite
└── run.py                             # Microservice supervisor launcher
```

---

## 6. Hardware Readiness & Acceleration Check

Hardware readiness was verified directly on the host machine:

```
======================================================================
🚀 HARDWARE DIAGNOSTICS & ACCELERATION CHECK
======================================================================
• Python Version        : 3.12.10
• PyTorch Version       : 2.6.0+cu124
• CUDA Available        : True
• Active Compute Device : cuda:0
• Dedicated GPU Name    : NVIDIA GeForce RTX 4060 Laptop GPU
• Total Dedicated VRAM  : 8.00 GB GDDR6
• Compute Capability    : sm_89 (Ada Lovelace Architecture)
• Tensor Core Precision : FP16 Automatic Mixed Precision (AMP)
• CUDA Tensor Test      : PASSED (Matrix multiplication verified on GPU)
======================================================================
```

---

## 7. Dependencies Audit (`requirements.txt`)

All required libraries for RT-DETR inference, BoT-SORT tracking, and PETS-2009 XML parsing were audited against the active `cv_env` environment:

| Dependency | Required Specification | Installed Version | Status & Capability |
| :--- | :--- | :--- | :--- |
| `torch` | `>=2.0.0` | `2.6.0+cu124` | **Ready** (CUDA 12.4 acceleration enabled) |
| `torchvision` | `>=0.15.0` | `0.21.0+cu124` | **Ready** (CUDA 12.4 enabled) |
| `ultralytics` | `>=8.3.0` | `8.4.142` | **Ready** (Native `RTDETR`, `YOLO`, `BOTSORT`, `ByteTrack`) |
| `lapx` | `>=0.5.5` | `0.9.4` (module: `lap`) | **Ready** (Linear assignment solver for BoT-SORT / ByteTrack) |
| `filterpy` | `>=1.4.5` | `1.4.5` | **Ready** (Kalman filtering for BoT-SORT) |
| `opencv-python-headless` | `>=4.8.0` | `5.0.0.93` (`cv2`) | **Ready** (High-throughput frame ingestion & drawing) |
| `xml.etree.ElementTree` | Standard Library | Python 3.12 builtin | **Ready** (PETS-2009 XML parsing) |
| `fastapi` | `>=0.110.0` | `0.141.1` | **Ready** (Asynchronous REST microservice) |
| `uvicorn` | `>=0.28.0` | `0.52.4` | **Ready** (ASGI server) |
| `python-multipart` | `>=0.0.9` | `0.0.32` | **Ready** (Video file upload streaming) |
| `streamlit` | `>=1.30.0` | `1.63.0` | **Ready** (Interactive analytics dashboard) |
| `plotly` | `>=5.18.0` | `7.0.0` | **Ready** (Real-time occupancy visualization) |
| `pandas` | `>=2.0.0` | `3.0.5` | **Ready** (Temporal timeseries aggregation) |
| `PyYAML` | `>=6.0` | `6.0.3` | **Ready** (Dataset & tracker configuration) |

---

## 8. Baseline Compatibility & Downstream RT-DETR Ingestion Status

### 1. Unified Detector Abstraction (`engine/detector.py`)
- `BaseDetector`: Abstract interface enforcing `predict()` and `track()` signatures.
- `YOLODetector`: Wraps `ultralytics.YOLO` for baseline models (`yolo11n.pt`, `runs/custom_train/best.pt`).
- `RTDETRDetector`: Wraps `ultralytics.RTDETR` for downstream domain adaptation. Supports fine-tuning and inference on RT-DETR variants (`rtdetr-r18.pt`, `rtdetr-l.pt`).
- `get_detector()`: Dynamic factory resolving architecture from weights path.

### 2. Multi-Object Tracking Abstraction (`engine/tracker.py`)
- `FootfallEngine` seamlessly accepts `tracker_config="botsort.yaml"` or `tracker_config="bytetrack.yaml"`.
- Automatically routes inference through `RTDETR` or `YOLO` without altering client interfaces or breaking existing API schemas.

### 3. Directional FSM Counter (`engine/fsm_counter.py`)
- Encapsulates `DualTripwireFSM`, `TrackState`, and `TrackInfo`.
- Supports horizontal gating (Y-axis), vertical gating (X-axis), and simultaneous dual-axis gating.
- Includes velocity-aware trajectory segment crossing with a 5-pixel margin buffer and hysteresis deadband.

### 4. Comparative Evaluation Suite (`evaluation/benchmark.py`)
- Provides `BenchmarkSuite` with automated side-by-side comparative benchmarking between:
  * Baseline 1: YOLO11n + ByteTrack
  * Baseline 2: YOLO11n (Fine-Tuned) + ByteTrack
  * Target Architecture: Domain-Adapted RT-DETR-R18 + BoT-SORT
- Measures: Average FPS, Min/Avg/Max Latency (ms), Peak VRAM (MB), MOT tracking stability, and counting accuracy against ground truth.

### 5. Verification Test Suite Results
```powershell
python -m unittest discover -s . -p "test_*.py"
```
**Outcome:**
- `Ran 31 tests in 8.760s`
- **Result: OK (0 errors, 0 failures)**

---

## 9. Next Steps for RT-DETR + BoT-SORT Pipeline

1. **Ingest PETS-2009 Benchmark Data**:
   Place incoming benchmark archives in `data/raw/` and execute `python data/scripts/convert_pets2009.py` to generate `data/processed/pets2009/` and manifest `data/processed/pets2009.yaml`.
2. **Domain Adaptation Training**:
   Fine-tune RT-DETR-R18 on `data/processed/pets2009.yaml` targeting surveillance gate camera geometry.
3. **Execute Comparative Benchmark**:
   Run `python evaluation/benchmark.py` to generate the formal side-by-side performance comparison against the preserved YOLO11n baseline.
