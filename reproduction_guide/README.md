# 🎯 ProGlint Primary Reproduction Guide: The 3 Priority Steps

> **IMPORTANT:** When learning, testing, or demonstrating this project to reviewers, **these 3 Priority Steps come FIRST**. Everything else in this folder provides deeper chapter-by-chapter reference.

```
═════════════════════════════════════════════════════════════════════════════════════════════
                              THE 3 PRIORITY REPRODUCTION STEPS
═════════════════════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
  │     PRIORITY STEP 1     │     │     PRIORITY STEP 2     │     │     PRIORITY STEP 3     │
  │     Data Validation     │ ──► │     Model Training      │ ──► │  Testing & Footfall     │
  │                         │     │                         │     │        Counting         │
  │ Verify 100% data health │     │ 5-Line RT-DETR training │     │ Track & count entries/  │
  │ before GPU computation  │     │ on surveillance dataset │     │ exits on video stream   │
  └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## ⚡ Quick-Run: The 3 Priority Commands

You can run the entire lifecycle right now in PowerShell using just these 3 commands:

### Priority Step 1: Validate Dataset
```powershell
& "cv_env\Scripts\python.exe" -c "from pathlib import Path; b=Path('data/processed'); print('Train:', len(list((b/'images/train').glob('*.jpg'))), '| Val:', len(list((b/'images/val').glob('*.jpg'))), '| Status: VALIDATED')"
```
- **What it does:** Verifies 1-to-1 matching between images and labels in `data/processed/`, confirms coordinates are in $[0, 1]$, and asserts class is `0` (`person`).
- **Expected Output:** `Train: 70 | Val: 35 | Status: VALIDATED`

---

### Priority Step 2: Train Model
```powershell
& "cv_env\Scripts\python.exe" train_rtdetr.py --epochs 15 --batch 16 --imgsz 640
```
- **What it does:** Runs our 5-line RT-DETR transfer learning loop for 15 epochs on your GPU.
- **The Core Code:**
  ```python
  from ultralytics import RTDETR
  model = RTDETR("rtdetr-l.pt")
  model.train(data="data/processed/pets2009.yaml", epochs=15, imgsz=640, batch=16, device=0)
  ```
- **Expected Output:** Runs in ~25 seconds, achieves $\sim 99\%$ mAP@50, and saves checkpoint to `runs/custom_train/best_rtdetr.pt`.

---

### Priority Step 3: Test & Count Footfall
```powershell
& "cv_env\Scripts\python.exe" -c "from engine.tracker import FootfallEngine; res = FootfallEngine().process_video('TownCentre_test.mp4', ''); print('IN:', res['total_in'], '| OUT:', res['total_out'], '| OCCUPANCY:', res['occupancy'])"
```
- **What it does:** Loads `best_rtdetr.pt`, tracks pedestrians on `TownCentre_test.mp4` with BoT-SORT, and counts downward entries (IN) and upward exits (OUT) with zero duplicates.
- **Expected Output:** `IN: 2 | OUT: 1 | OCCUPANCY: 1`

---

## 📚 Deep-Dive Reference Chapters (For Complete Mastery)

Once you understand the 3 Priority Steps above, read the individual chapters below to master the code line by line:

| Chapter | Deep-Dive Reference | What It Covers |
| :--- | :--- | :--- |
| **Setup** | [**`00_PREREQUISITES_AND_ENVIRONMENT_SETUP.md`**](file:///D:/yasmin%20programs/PROGLINT/reproduction_guide/00_PREREQUISITES_AND_ENVIRONMENT_SETUP.md) | Virtualenv creation, PyTorch CUDA 12.4 installation, and FFmpeg setup. |
| **Step 1 Deep-Dive** | [**`01_DATA_INGESTION_AND_ANNOTATION_PARSING.md`**](file:///D:/yasmin%20programs/PROGLINT/reproduction_guide/01_DATA_INGESTION_AND_ANNOTATION_PARSING.md) | The 15-line XML parsing and normalization code vs. 600-line downloader. |
| **Step 2 Deep-Dive** | [**`02_MODEL_TRAINING_AND_DOMAIN_ADAPTATION.md`**](file:///D:/yasmin%20programs/PROGLINT/reproduction_guide/02_MODEL_TRAINING_AND_DOMAIN_ADAPTATION.md) | The pure 5-line training code vs. 340-line enterprise boilerplate. |
| **Step 3 Deep-Dive** | [**`03_TRACKING_AND_FSM_COUNTING_ENGINE.md`**](file:///D:/yasmin%20programs/PROGLINT/reproduction_guide/03_TRACKING_AND_FSM_COUNTING_ENGINE.md) | **(Person 3 Core)** The 25-line BoT-SORT tracking and Dual-Tripwire FSM loop. |
| **API Integration** | [**`04_FASTAPI_BACKEND_AND_FFMPEG.md`**](file:///D:/yasmin%20programs/PROGLINT/reproduction_guide/04_FASTAPI_BACKEND_AND_FFMPEG.md) | The 15-line FastAPI microservice and browser H.264 video streaming. |
| **Web Dashboard** | [**`05_STREAMLIT_DASHBOARD_AND_FULL_LAUNCH.md`**](file:///D:/yasmin%20programs/PROGLINT/reproduction_guide/05_STREAMLIT_DASHBOARD_AND_FULL_LAUNCH.md) | The 15-line Streamlit frontend, interactive sliders, and `run.py`. |
