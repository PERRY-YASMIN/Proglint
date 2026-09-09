# Document 1: End-to-End System Architecture & The 3-Person Pipeline

## 1. Welcome to ProGlint: What Is This Project?

**ProGlint Footfall Intelligence** is a production-grade, edge-deployable Computer Vision (CV) system designed to monitor human traffic through entryways, corridors, turnstiles, and security gates using monocular closed-circuit television (CCTV) cameras.

### The Real-World Problem We Solve
Traditional retail and surveillance installations attempt to count people using:
1. **Infrared break-beams:** These sensors emit a beam across a doorway. If two people walk in together shoulder-to-shoulder, only one break is registered (under-counting). If someone stands on the beam waving their hands, it counts multiple times (over-counting). They also cannot tell whether someone entered or exited.
2. **Standard Computer Vision (YOLO + Single Virtual Line):** When an off-the-shelf YOLO model is used, people standing near the doorway oscillate back and forth across the line due to bounding box jitter, creating dozens of fake counts. Furthermore, in dense crowds viewed from overhead CCTV cameras, YOLO's Non-Maximum Suppression (NMS) collapses, deleting overlapping pedestrians.

### The ProGlint Solution
ProGlint solves this with a modern 3-stage intelligence pipeline:
- **Detection (RT-DETR):** A Vision Transformer that detects humans as a direct set prediction, completely eliminating NMS box collapse.
- **Tracking (BoT-SORT):** Assigns persistent identity numbers (Track IDs) to each person across consecutive video frames, compensating for camera jitter and brief occlusions.
- **Accounting (Dual-Tripwire Finite State Machine):** Uses bottom-center foot coordinates and a spatial deadband between two virtual lines to guarantee zero duplicate counts, direction verification, and real-time net occupancy.

```
═════════════════════════════════════════════════════════════════════════════════════════════════════
                                   THE COMPLETE PROGLINT PIPELINE
═════════════════════════════════════════════════════════════════════════════════════════════════════

  [ CCTV Video Stream / File ]
              │
              ▼
  ┌────────────────────────────────────────┐
  │ 👤 PERSON 1: Detection Transformer     │
  │ • Model: Domain-Adapted RT-DETR-R18    │
  │ • Input: 640x640 RGB Image Tensor      │
  │ • Output: Bounding Boxes [x1,y1,x2,y2] │
  └───────────────────┬────────────────────┘
                      │ Detections [x1, y1, x2, y2, conf, cls=0]
                      ▼
  ┌────────────────────────────────────────┐
  │ 👤 PERSON 2: Multi-Object Tracking     │
  │ • Tracker: BoT-SORT (botsort.yaml)     │
  │ • State: Kalman Filter + Re-ID + CMC   │
  │ • Output: Persistent Track IDs (tid)   │
  └───────────────────┬────────────────────┘
                      │ Active Track IDs + Boxes: (tid, [x1, y1, x2, y2])
                      ▼
  ┌────────────────────────────────────────┐
  │ 👤 PERSON 3: Accounting & Integration  │
  │ • Geometry: Foot Point (cx, cy = y2)   │
  │ • Logic: Dual-Tripwire FSM (A and B)   │
  │ • Guard: State Lock + Timeout + Memory │
  │ • Output: Total IN, OUT, Net Occupancy │
  └───────────────────┬────────────────────┘
                      │
                      ├──────────────────────────────────────────┐
                      ▼                                          ▼
  ┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
  │ FastAPI Backend (backend/main.py)      │ │ Streamlit Frontend (frontend/app.py)   │
  │ • POST /process_video                  │ │ • Interactive Tripwire Sliders         │
  │ • GET /health Telemetry                │ │ • Video Ingestion & HTML5 Player       │
  │ • FFmpeg H.264 Ultrafast Transcoding   │ │ • Plotly Occupancy Step-Charts         │
  └────────────────────────────────────────┘ └────────────────────────────────────────┘
```

---

## 2. The 3-Person Team Division of Responsibilities

To design, train, and deploy an enterprise CV system, the engineering workload is partitioned into three specialized domains:

### 👤 Person 1: Dataset Engineering & RT-DETR Detection
- **Core Question:** *"Where are the people in this single frame right now?"*
- **Primary Modules:**
  - `data/scripts/convert_pets2009.py` (CVML XML parser & dataset partitioner)
  - `data/processed/pets2009.yaml` (Dataset configuration manifest)
  - `train_rtdetr.py` (Ultralytics RT-DETR fine-tuning script)
  - `runs/custom_train/best_rtdetr.pt` (Trained model weights)
- **What they must explain:**
  1. What dataset is used (PETS-2009 access-control benchmark).
  2. How annotations are parsed from XML and converted into normalized YOLO format `[0, xc, yc, w, h]`.
  3. Why frame-by-frame random splitting causes data leakage in video, and how sequence-based splitting solves it.
  4. What RT-DETR (Real-Time Detection Transformer) is and why it replaces CNNs (YOLO).
  5. How transfer learning was executed (AdamW, FP16, 640x640 resolution, 15 epochs).
  6. The validation metrics achieved ($0.9949$ mAP@50).

### 👤 Person 2: Multi-Object Tracking & BoT-SORT
- **Core Question:** *"Which person from the previous frame corresponds to which person in the current frame?"*
- **Primary Modules:**
  - `engine/tracker.py` (`model.track(..., tracker="botsort.yaml")`)
  - Integration with Ultralytics BoT-SORT C++ / Python tracking modules.
- **What they must explain:**
  1. The distinction between object detection (memoryless frame snapshots) and tracking (temporal identity).
  2. What a Track ID is and why raw bounding boxes alone cannot compute trajectories.
  3. How detection boxes are formatted and passed to the tracker.
  4. The 3 pillars of BoT-SORT:
     - **Kalman Filter:** Estimates velocity and predicts position in the next frame.
     - **Camera Motion Compensation (CMC):** Subtracts camera shake using optical flow homography.
     - **Appearance Re-ID:** Matches deep feature embeddings during occlusions.
  5. How tracks are classified as active, lost, or recovered.
  6. Foot-point extraction: Why the bottom-center coordinate $(x_c, y_2)$ is used instead of the bounding box centroid $(x_c, y_c)$ due to perspective distortion.

### 👤 Person 3: Directional Counting, FSM & System Integration
- **Core Question:** *"Did this tracked person physically cross our entry threshold, in what direction, and how does that update our live audit log?"*
- **Primary Modules:**
  - `engine/tracker.py` (`FootfallEngine`, FSM state logic, HUD rendering)
  - `backend/main.py` (FastAPI REST service, video file management, FFmpeg transcoding)
  - `frontend/app.py` (Streamlit web interface, calibration sliders, Plotly analytics)
  - `test_footfall_engine.py` & `test_e2e.py` (Automated unit and integration test suites)
- **What they must explain:**
  1. Image coordinate conventions ($(0,0)$ at top-left, $Y$ downwards).
  2. The mathematical definition of virtual tripwires (Line A at $0.45\cdot H$, Line B at $0.55\cdot H$) and the spatial hysteresis deadband.
  3. Why a single line fails and how a dual-line turnstile eliminates loitering/hesitation false counts.
  4. The Finite State Machine (FSM): `IDLE` $\rightarrow$ `PENDING_IN` / `PENDING_OUT` $\rightarrow$ `DONE`.
  5. Duplicate prevention mechanisms: State locking, motion displacement check ($>15\text{ px}$), and temporal timeout ($3.0\text{ s}$).
  6. Real-time net occupancy calculation: $\text{Occupancy} = \text{Total IN} - \text{Total OUT}$.
  7. Active memory garbage collection: Purging inactive track histories after 60 frames to prevent RAM leaks.
  8. Concrete walk-through of physical scenarios (Person 7 $\rightarrow$ IN, Person 8 $\rightarrow$ OUT, Person 9 $\rightarrow$ IN).
  9. System integration: Decoupling the engine into a FastAPI microservice and interactive Streamlit UI.

---

## 3. Project Directory Map & Execution Entry Points

Here is how every folder and critical file is organized on disk:

```
D:\yasmin programs\PROGLINT\
├── data/
│   ├── raw/                             <-- Raw downloaded PETS-2009 datasets and XMLs
│   ├── processed/                       <-- Formatted images/ & labels/ for RT-DETR
│   │   ├── images/train/, images/val/   <-- Extracted video frames (.jpg)
│   │   ├── labels/train/, labels/val/   <-- Normalized YOLO text files (.txt)
│   │   └── pets2009.yaml                <-- Training manifest consumed by Ultralytics
│   └── scripts/
│       └── convert_pets2009.py          <-- Dataset conversion and sequence splitter
├── engine/
│   ├── __init__.py
│   └── tracker.py                       <-- Consolidated FootfallEngine (RT-DETR + BoT-SORT + FSM)
├── backend/
│   ├── __init__.py
│   └── main.py                          <-- FastAPI REST API service
├── frontend/
│   └── app.py                           <-- Streamlit interactive web dashboard
├── evaluation/
│   ├── benchmark.py                     <-- Head-to-head empirical benchmark script
│   └── BENCHMARK_REPORT.md              <-- Latency, FPS, and tracking metrics report
├── runs/
│   ├── custom_train/best_rtdetr.pt      <-- Our fine-tuned surveillance model checkpoint
│   └── rtdetr_train/                    <-- Training logs, confusion matrices, loss curves
├── rtdetr-l.pt                          <-- Base COCO pre-trained RT-DETR weights (Baidu/Ultralytics)
├── TownCentre_test.mp4                  <-- 140-frame CCTV surveillance test clip
├── train_rtdetr.py                      <-- End-to-end model fine-tuning script
├── run.py                               <-- Dual-process launcher (FastAPI + Streamlit concurrently)
└── test_footfall_engine.py              <-- Unit test suite for FSM and counting logic
```

### How to Launch the Entire System

You can run the full system using the launcher script:
```powershell
python run.py
```
This automatically starts:
1. **The Backend REST Server:** Running on `http://127.0.0.1:8000` via Uvicorn.
2. **The Frontend Web UI:** Running on `http://localhost:8501` via Streamlit.

---

## 4. End-to-End Lifecycle of a Video Frame

To visualize how data flows through the application, trace what happens to a single frame:

```
1. DISK / UPLOAD:
   User uploads TownCentre_test.mp4 on Streamlit UI (port 8501).
   Streamlit sends the video bytes via HTTP POST to http://127.0.0.1:8000/process_video.

2. OPENCV DECODING:
   FastAPI receives the file and invokes FootfallEngine.process_video().
   OpenCV cv2.VideoCapture reads Frame #45 as a NumPy array (shape: [1080, 1920, 3], dtype: uint8).

3. DETECTOR INFERENCE (Person 1):
   Frame is passed to RTDETR.track(..., imgsz=640, conf=0.40, tracker="botsort.yaml").
   RT-DETR resizes the image to 640x640, converts to tensor, and executes transformer forward pass.
   Outputs a list of predicted bounding boxes: [x1, y1, x2, y2] with confidence >= 0.40.

4. TRACKER DATA ASSOCIATION (Person 2):
   BoT-SORT takes the boxes, compensates for camera motion using optical flow homography,
   predicts person positions using Kalman filter states, and assigns persistent Track IDs:
   e.g., Track ID 7 is detected at box [480, 220, 560, 410].

5. FOOT-POINT & FSM EVALUATION (Person 3):
   Bottom-center coordinate is calculated: cx = (480 + 560)/2 = 520, cy = 410.
   FootfallEngine checks cy against Line A (y=486) and Line B (y=594).
   FSM updates Person 7's state from IDLE -> PENDING_IN.
   HUD annotations (boxes, green ID label, magenta/cyan tripwire lines, count banner) are drawn with cv2.

6. H.264 TRANSCODING & CLIENT PLAYBACK:
   Processed frames are written to an MP4 video file.
   FFmpeg transcodes the file into web-native H.264 (yuv420p).
   FastAPI returns JSON metrics: {"total_in": 1, "total_out": 0, "occupancy": 1, ...}.
   Streamlit updates the live dashboard and plays the processed video in the browser.
```
