# ProGlint Footfall Intelligence

Production-grade real-time computer vision system for directional pedestrian tracking, access-control gate monitoring, and footfall analytics.

---

## Architectural Transition: RT-DETR + BoT-SORT

ProGlint is transitioning from the baseline **YOLO11n + ByteTrack** architecture to a domain-adapted **RT-DETR (RT-DETR-R18) + BoT-SORT** pipeline fine-tuned on the **PETS-2009** surveillance benchmark.

- **Detection**: RT-DETR (Real-Time DEtection TRansformer) eliminates NMS bottlenecks and delivers superior bounding box precision in dense CCTV crowds without modifying internal transformer neural architecture.
- **Tracking**: BoT-SORT combines Kalman filter motion estimation with camera motion compensation (CMC) and appearance feature re-identification, significantly reducing ID switching during occlusions.
- **Access Control**: Dual-tripwire directional Finite State Machine (FSM) with velocity-aware trajectory segment crossing, 5-pixel margin buffers, and hysteresis deadband.
- **Baseline Preservation**: The YOLO11n + ByteTrack pipeline is preserved for comparative evaluation and benchmarking.

---

## Standardized Project Directory Structure

```
PROGLINT/
├── data/
│   ├── raw/                 # Incoming benchmark archives (PETS-2009 XMLs & images)
│   ├── processed/           # Parsed YOLO-format annotations & YAML manifests
│   └── scripts/             # Conversion & dataset preparation scripts
├── engine/
│   ├── __init__.py          # Public package interface
│   ├── detector.py          # Unified detector abstraction (RT-DETR & YOLO baseline)
│   ├── tracker.py           # Multi-object tracker module (BoT-SORT & ByteTrack)
│   └── fsm_counter.py       # Dual-tripwire directional Finite State Machine
├── backend/
│   ├── __init__.py
│   └── main.py              # FastAPI microservice
├── frontend/
│   └── app.py               # Streamlit analytics dashboard
├── evaluation/
│   ├── __init__.py
│   └── benchmark.py         # Comparative evaluation suite (mAP, HOTA, counting accuracy)
├── runs/                    # Dynamic artifacts (weights, logs, transcode caches)
│   └── custom_train/        # Preserved trained model weights
├── requirements.txt         # Core dependencies
└── README.md                # Project documentation
```

---

## Hardware Readiness & Acceleration

The system is optimized for NVIDIA Ada Lovelace architecture:
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB GDDR6 VRAM)
- **CUDA Runtime**: CUDA 12.4
- **PyTorch**: 2.6.0+cu124
- **Precision**: FP16 Automatic Mixed Precision (AMP) enabled

---

## Quick Start

### 1. Environment Setup

Activate the virtual environment:
```powershell
.\cv_env\Scripts\activate
```

Install or update dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Run Microservice & Dashboard

Run both backend and frontend concurrently:
```powershell
python run.py both
```

Or run independently:
```powershell
# FastAPI Backend on http://127.0.0.1:8000
python run.py backend

# Streamlit UI on http://127.0.0.1:8501
python run.py frontend
```

### 3. Run Comparative Benchmark Suite

Benchmark the baseline YOLO11n pipeline side-by-side with RT-DETR + BoT-SORT:
```powershell
python evaluation/benchmark.py --video TownCentre_test.mp4 --max_frames 50
```

### 4. PETS-2009 Dataset Preparation

Extract raw PETS-2009 archives and generate formatted manifests:
```powershell
python data/scripts/convert_pets2009.py --raw_dir data/raw --processed_dir data/processed/pets2009
```

---

## Test Suite Execution

Run all unit, API, and end-to-end integration tests:
```powershell
# Engine & FSM unit tests
python -m unittest test_footfall_engine.py

# Backend API & End-to-End pipeline tests
python -m unittest test_backend_api.py test_e2e.py
```
