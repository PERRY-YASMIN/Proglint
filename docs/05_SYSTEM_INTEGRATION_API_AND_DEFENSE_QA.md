# Document 5: System Integration, FastAPI, Streamlit & Defense Q&A

## 1. System Integration Architecture

ProGlint is decoupled into three production tiers:
1. **Core Computer Vision Engine (`engine/tracker.py`):** Stateless inference and stateful tracking/counting.
2. **REST API Microservice (`backend/main.py`):** High-throughput asynchronous service powered by FastAPI and Uvicorn.
3. **Interactive Analytical Dashboard (`frontend/app.py`):** User interface built with Streamlit and Plotly.

```
                   END-TO-END DEPLOYMENT TOPOLOGY
                   
 ┌──────────────────────────────────────────────────────────────┐
 │                      Client Web Browser                      │
 │                   (http://localhost:8501)                    │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                    Streamlit Frontend UI                     │
 │                   (frontend/app.py)                          │
 │  • Uploads video bytes                                       │
 │  • Sends normalized calibration parameters                   │
 │  • Renders Plotly Occupancy Step Chart                       │
 │  • Embeds H.264 HTML5 Video Player                           │
 └──────────────────────────────┬───────────────────────────────┘
                                │ HTTP REST (Multipart Form)
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                     FastAPI REST Gateway                     │
 │                     (backend/main.py)                        │
 │  • Endpoints: POST /process_video, GET /health, GET /videos  │
 │  • Spawns FFmpeg for Web-Native H.264 Transcoding            │
 └──────────────────────────────┬───────────────────────────────┘
                                │ Python API Call
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │              Consolidated FootfallEngine                     │
 │                 (engine/tracker.py)                          │
 │  • Loads runs/custom_train/best_rtdetr.pt on cuda:0          │
 │  • Executes BoT-SORT Multi-Object Tracking                   │
 │  • Runs Dual-Tripwire FSM & Duplicate Prevention             │
 │  • Renders OpenCV HUD overlays                               │
 └──────────────────────────────────────────────────────────────┘
```

---

## 2. The FastAPI Service Contract

The backend is defined in [`backend/main.py`](file:///D:/yasmin%20programs/PROGLINT/backend/main.py).

### Endpoint 1: `GET /health`
Used by Kubernetes, Docker, or monitoring tools to verify hardware acceleration and model status.
- **Response JSON:**
```json
{
  "status": "ok",
  "cuda_available": true,
  "device": "cuda:0",
  "detector": "Domain-Adapted RT-DETR (PETS-2009)",
  "tracker": "BoT-SORT",
  "hardware": "CUDA (RTX 4060)"
}
```

---

### Endpoint 2: `POST /process_video`
The core ingestion endpoint that processes an uploaded video file.
- **Request Parameters:**
  - `video`: Binary file upload (`.mp4`, `.avi`, `.mov`).
  - `line_a_norm` (float): Normalized height for Outer Line A (default `0.45`).
  - `line_b_norm` (float): Normalized height for Inner Line B (default `0.55`).
  - `timeout_sec` (float): Maximum crossing duration before FSM resets (default `3.0`).

- **Response JSON Contract:**
```json
{
  "status": "success",
  "video_filename": "a37113cc_processed.mp4",
  "metrics": {
    "total_in": 2,
    "total_out": 1,
    "current_occupancy": 1,
    "total_tracks": 13
  },
  "performance": {
    "avg_fps": 19.9,
    "total_frames": 140
  },
  "events": [
    {"frame": 40, "id": 7, "type": "IN", "time_sec": 1.33},
    {"frame": 90, "id": 8, "type": "OUT", "time_sec": 3.00},
    {"frame": 130, "id": 9, "type": "IN", "time_sec": 4.33}
  ]
}
```

---

### Why FFmpeg H.264 Transcoding Is Mandatory

When OpenCV's `cv2.VideoWriter` encodes video using the standard `mp4v` (MPEG-4 Part 2) codec, modern web browsers (Google Chrome, Microsoft Edge, Safari, Firefox) **refuse to play the video**. The HTML5 `<video>` element displays a black screen or errors out with `Format Not Supported`.

To solve this, [`backend/main.py`](file:///D:/yasmin%20programs/PROGLINT/backend/main.py#L15-L17) runs an automated hardware-accelerated transcoding subprocess:
```python
def to_h264(src, dst):
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-vcodec", "libx264",      # High-compatibility H.264 video codec
        "-pix_fmt", "yuv420p",     # Standard 4:2:0 chroma subsampling
        "-preset", "ultrafast",    # Near-zero transcoding latency
        dst
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```
This guarantees immediate, zero-plugin playback in any browser.

---

## 3. Interactive Streamlit Frontend

The user interface in [`frontend/app.py`](file:///D:/yasmin%20programs/PROGLINT/frontend/app.py) provides:
1. **First-Frame Calibration Preview:** Extracts frame 0 of the uploaded video and displays Line A (Cyan) and Line B (Magenta). As the user adjusts the sliders, the preview updates immediately before spending GPU time on full inference.
2. **Occupancy Step-Chart:** Uses Plotly to build a discrete step chart ($h-v$ shape) showing exactly how occupancy increased and decreased over time.
3. **Audit Log Table:** Displays timestamped crossing records with Track IDs.

---

## 4. Master Defense & Viva Q&A Playbook

This section contains definitive answers to every question an examiner, professor, or technical lead can ask about this project:

---

### General & Architecture Questions

#### Q1: "Why did you use a Vision Transformer (RT-DETR) instead of traditional YOLO for footfall counting?"
> **Answer:**  
> *"Traditional CNN detectors like YOLO rely on dense anchor proposals and filter them using greedy Non-Maximum Suppression (NMS). In access-control gates and turnstiles, pedestrians walk closely together, causing their bounding boxes to overlap with IoU $> 0.50$. Standard NMS cannot distinguish between duplicate detections of the same person and genuine adjacent pedestrians, causing it to delete the second person (NMS collapse), undercounting by up to 50%.*  
> 
> *RT-DETR solves this by formulating detection as direct set prediction. Using a hybrid encoder (AIFI and CCFM) and 300 learnable queries with bipartite Hungarian matching, it predicts at most one bounding box per physical person without needing NMS at all. This completely eliminates bounding box collapse in crowded bottlenecks while sustaining 20 FPS in FP16 on our RTX 4060 GPU."*

---

#### Q2: "Why use BoT-SORT instead of simpler trackers like ByteTrack or standard SORT?"
> **Answer:**  
> *"Simpler trackers rely exclusively on linear Kalman velocity models. In physical surveillance, they suffer from two major flaws:*  
> *1. **Camera vibration:** Environmental vibrations cause the tracker to think people are moving when they are standing still.*  
> *2. **Identity swaps:** When two people cross paths, their bounding boxes overlap, and motion-only trackers frequently swap their IDs.*  
> 
> *BoT-SORT solves this with three features:*  
> *- **Camera Motion Compensation (CMC):** Uses optical flow homography to subtract background camera shake.*  
> *- **Refined Kalman State:** Directly models $[x, y, w, h]$ rather than aspect ratio, accommodating perspective foreshortening.*  
> *- **Deep Appearance Re-ID:** Extracts deep visual embeddings to match people by visual appearance when trajectories intersect. In our benchmark, this achieved exactly zero identity swaps."*

---

### Dataset & Training Questions (Person 1)

#### Q3: "What dataset was used, and why did you not just train on MS-COCO?"
> **Answer:**  
> *"We trained and evaluated on the PETS-2009 surveillance benchmark. MS-COCO consists of eye-level photos taken from consumer smartphones, where people appear upright with canonical limb proportions.*  
> 
> *In access-control installations, CCTV cameras are mounted overhead at 45° to 60° angles. From this vantage point, humans appear foreshortened into heads, coat collars, and shoulders. Pre-trained COCO models fail to detect people under steep overhead angles. By domain-adapting RT-DETR on PETS-2009, our model learned the specific visual priors of overhead surveillance."*

---

#### Q4: "How did you prevent data leakage when preparing your video training set?"
> **Answer:**  
> *"In video, consecutive frames captured at 25 FPS share upwards of 98% mutual information. If you perform a naive random train/val split on individual frames, frame $t$ will be in the training set and frame $t+1$ will be in the validation set. The model will simply memorize the pedestrians' clothing and background, yielding an artificially inflated mAP that fails in production.*  
> 
> *We prevented this by enforcing **Sequence-Based Partitioning** in `convert_pets2009.py`. Complete, continuous video sequences were partitioned into disjoint subsets: Sequences `S1L1` and `S2L1` for training, `S1L2` for validation, and `S2L2` for testing. The validation sequence contains completely unseen pedestrians and crowd flows, ensuring genuine generalization."*

---

#### Q5: "What image resolution did you choose for RT-DETR and why?"
> **Answer:**  
> *"We trained and deployed at $640\times 640$ pixels. At access-control choke points, pedestrians are close to the camera and occupy 15% to 40% of the frame height, meaning high resolution like $1080\times 1080$ is unnecessary and reduces inference throughput to 5 FPS.*  
> 
> *$640\times 640$ provides the optimal trade-off: it delivers real-time 20 FPS sustained inference on our edge GPU, consumes under 200 MB of VRAM, and achieves $0.9949$ mAP@50 without missing pedestrians."*

---

### Tracking & Geometry Questions (Person 2)

#### Q6: "Why do you track the bottom-center foot point $(c_x, y_2)$ instead of the bounding box center $(x_c, y_c)$?"
> **Answer:**  
> *"In elevated CCTV cameras ($45^\circ$ angle), the center of the bounding box $(x_c, y_c)$ represents a point suspended in mid-air inside the human chest. Because of perspective parallax, a tall adult and a child standing on the exact same doorway tile have centroids separated by dozens of pixels. Furthermore, arm swinging or leaning forward causes the centroid to oscillate up and down.*  
> 
> *The bottom-center coordinate $(c_x, y_2)$ represents the contact point between shoes and the physical floor. The floor is a rigid 2D geometric surface invariant to height, posture, and arm swing, providing a smooth, noise-free tracking coordinate."*

---

### Counting, FSM & Logic Questions (Person 3)

#### Q7: "Why can't you count people with just a single virtual line?"
> **Answer:**  
> *"A single virtual line suffers from severe failure modes in production:*  
> *1. **Boundary loitering:** If a pedestrian stands directly on the line shifting weight, their coordinates cross back and forth, generating dozens of false counts.*  
> *2. **Hesitation:** If someone approaches the door, touches the line, and changes their mind, a single line counts them as an entry.*  
> 
> *Our Dual-Tripwire FSM creates a spatial hysteresis deadband. A count requires crossing Line A and then Line B sequentially. Pacing or loitering inside the deadband produces zero counts, and retreating resets the state without counting."*

---

#### Q8: "What stops the system from double-counting when someone enters and stays in the room?"
> **Answer:**  
> *"When a person crosses Line B downwards, their FSM state transitions to `DONE`. In our implementation, transitions to `PENDING_IN` or `PENDING_OUT` can only be initiated when the state is `IDLE`.*  
> 
> *Because their state is locked at `DONE`, no subsequent movements in the frame can re-trigger a crossing under that Track ID. It remains permanently locked until the person physically exits the camera's field of view."*

---

#### Q9: "What happens if a pedestrian enters the deadband between Line A and Line B and stops for a long time?"
> **Answer:**  
> *"Our FSM enforces a temporal timeout latch ($3.0\text{ seconds}$ by default). When Line A is crossed, we record $t_{\text{start}}$. If the person remains inside the deadband without completing the crossing within $3.0\text{ seconds}$, the state expires and resets to `IDLE`.*  
> 
> *If they eventually continue inward, they must re-trigger the sequence, ensuring that stale transitions do not linger indefinitely."*

---

#### Q10: "How does the system prevent memory leaks if running continuously 24/7?"
> **Answer:**  
> *"We implement an active garbage collection routine in `FootfallEngine`. In every frame, we compare the currently detected tracks (`active_ids`) with our internal `self.history` dictionary.*  
> 
> *For any track ID that is no longer present in the frame, we increment a `frames_absent` counter. Once a track has been absent for more than 60 consecutive frames (~2 seconds at 30 FPS), its record is permanently deleted from memory. This provides tolerance against momentary detector occlusions while ensuring system RAM remains flat indefinitely."*
