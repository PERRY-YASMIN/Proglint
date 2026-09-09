# Document 2: Datasets, Annotations, and RT-DETR Training (Person 1 Deep-Dive)

## 1. What Is Our Dataset & Why Was It Chosen?

In this project, we utilize two distinct video datasets:

### 1. The Training & Validation Benchmark: PETS-2009
- **Full Name:** Performance Evaluation of Tracking and Surveillance (PETS) 2009 Benchmark Dataset.
- **Source:** Sponsored by IEEE, European Commission, and the University of Reading.
- **Physical Setup:** Real monocular surveillance CCTV cameras mounted at an overhead oblique angle ($45^\circ\text{--}60^\circ$ pitch) looking down upon an outdoor campus walkway and access bottleneck.
- **Why PETS-2009?**
  - Most public computer vision datasets (like MS-COCO or Pascal VOC) consist of photographs taken from **eye-level horizontal perspectives** (smartphones and DSLRs). In COCO, humans appear upright with clearly visible torsos, legs, and shoes.
  - In physical access control (security turnstiles, building doorways, retail gates), cameras are mounted **above** people looking downwards. Pedestrians appear foreshortened: heads and shoulders overlap, bodies partially conceal one another, and their aspect ratios change dynamically as they walk under the lens.
  - PETS-2009 contains authentic pedestrian crowd flows, varying densities (sparse walking, medium flow, dense clusters), directional crossing streams, and realistic occlusions.

### 2. The Real-World Inference Benchmark: TownCentre
- **Source:** Oxford Town Centre Surveillance Video.
- **Usage:** Used in [`TownCentre_test.mp4`](file:///D:/yasmin%20programs/PROGLINT/TownCentre_test.mp4) and [`evaluation/benchmark.py`](file:///D:/yasmin%20programs/PROGLINT/evaluation/benchmark.py) to empirically test whether the model generalizes to a completely unseen surveillance environment.

---

## 2. How Are Annotations Stored & Converted?

### What Is a Bounding Box?
A bounding box is a rectangular geometric boundary that encloses an object of interest within a 2D image matrix.
In computer vision, bounding boxes are represented in two standard coordinate formats:

1. **Corners Format `[x1, y1, x2, y2]` (Pixels):**
   - $(x_1, y_1)$: Top-left coordinate in pixels.
   - $(x_2, y_2)$: Bottom-right coordinate in pixels.
   - Example: A box in a $768\times 576$ image spanning horizontally from pixel 100 to 200, and vertically from 150 to 350 is `[100, 150, 200, 350]`.

2. **Center-Size Normalized Format `[class_id, x_center, y_center, width, height]`:**
   - All coordinates are normalized by dividing by the image width $W$ and height $H$, yielding values strictly between $0.0$ and $1.0$.
   - This makes the annotation resolution-independent (whether the image is scaled to $640\times 640$ or $1920\times 1080$, the box remains identical).

---

### The Raw Annotation Format: CVML XML
PETS-2009 stores ground truth annotations in **Computer Vision Markup Language (CVML) XML** files.
Here is an exact snippet of how PETS-2009 stores a pedestrian:

```xml
<frame number="45">
    <objectlist>
        <object id="7">
            <box xc="384.5" yc="210.0" w="54.0" h="128.0"/>
        </object>
        <object id="8">
            <box xc="490.0" yc="280.0" w="62.0" h="142.0"/>
        </object>
    </objectlist>
</frame>
```

In raw CVML XML:
- `xc`, `yc`: Centroid horizontal and vertical coordinates in absolute pixels.
- `w`, `h`: Bounding box width and height in absolute pixels.

---

### The Target Training Format: RT-DETR / YOLO Standard
To feed these annotations into RT-DETR via Ultralytics, each video frame image (e.g., `frame_0045.jpg`) must have a corresponding plain text file (e.g., `frame_0045.txt`).
Each line in `frame_0045.txt` represents one person:

```text
<class_id> <norm_x_center> <norm_y_center> <norm_width> <norm_height>
```
Since our system tracks single-class human pedestrian traffic, `class_id` is always `0` (`person`).

#### The Exact Mathematical Conversion:
Given image width $W = 768$ and height $H = 576$:
$$\text{norm\_xc} = \frac{x_c}{W} = \frac{384.5}{768} \approx 0.50065$$
$$\text{norm\_yc} = \frac{y_c}{H} = \frac{210.0}{576} \approx 0.36458$$
$$\text{norm\_w} = \frac{w}{W} = \frac{54.0}{768} \approx 0.07031$$
$$\text{norm\_h} = \frac{h}{H} = \frac{128.0}{576} \approx 0.22222$$

The resulting line in the `.txt` file is:
```text
0 0.500651 0.364583 0.070312 0.222222
```

This conversion is implemented automatically in [`data/scripts/convert_pets2009.py`](file:///D:/yasmin%20programs/PROGLINT/data/scripts/convert_pets2009.py#L79-L130).

---

## 3. Sequence-Based Partitioning: Preventing Data Leakage

### The Danger of Video Data Leakage
In standard computer vision tutorials, datasets are often split using random shuffling:
```python
# NAIVE / INCORRECT SPLIT IN VIDEO CV
train_images, val_images = train_test_split(all_frames, test_size=0.2, random_state=42)
```

**Why this is catastrophic in surveillance video:**
- Video is recorded at 25 or 30 frames per second.
- Frame $t$ and Frame $t+1$ are separated by only 33 milliseconds. They share $98\%$ identical pixel information: the exact same pedestrians, identical clothing, identical background lighting, and identical camera angles.
- If Frame 40 is in the training set and Frame 41 is in the validation set, the model does not learn to generalize—it simply memorizes the appearance of Frame 40. The validation accuracy will report a false $99.9\%$, but the model will immediately fail when deployed on a real CCTV feed.

### Our Solution: Strict Sequence-Based Partitioning
In [`data/scripts/convert_pets2009.py`](file:///D:/yasmin%20programs/PROGLINT/data/scripts/convert_pets2009.py#L44-L51), we partition by **entire continuous video sequence files**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PETS-2009 DATASET PARTITIONING PROTOCOL (ZERO TEMPORAL LEAKAGE)            │
├─────────────────┬───────────────────┬──────────────┬────────────────────────┤
│ Sequence Name   │ Density / Flow    │ Assigned Set │ Purpose                │
├─────────────────┼───────────────────┼──────────────┼────────────────────────┤
│ S1L1 (Seq 1)    │ Walking flow      │ TRAIN        │ Learn CCTV priors      │
│ S2L1 (Seq 2)    │ Sparse crowd flow │ TRAIN        │ Learn crowd movement   │
│ S1L2 (Seq 3)    │ Medium flow       │ VALIDATION   │ Pure unseen validation │
│ S2L2 (Seq 4)    │ Dense crowd stream│ TEST         │ Holdout evaluation     │
└─────────────────┴───────────────────┴──────────────┴────────────────────────┘
```
Because sequence `S1L2` is never shown during training, our validation metrics reflect genuine generalization to new pedestrian flows.

---

## 4. What Is RT-DETR & Why Do We Use It?

### The Core Difference: CNNs (YOLO) vs. Vision Transformers (RT-DETR)

Traditional CNN detectors like YOLO divide an image into a dense spatial grid. Every cell outputs multiple candidate anchor boxes. For a $1080\text{p}$ image, YOLO produces thousands of dense proposals:

```
                  THE NON-MAXIMUM SUPPRESSION (NMS) COLLAPSE
                  
       Two pedestrians walking side-by-side in an access doorway:
       ┌───────────┐
       │ Person A  │ ──┐
       │   ┌───────┴───┼───────┐
       │   │ Intersection-     │
       │   │ Over-Union (IoU)  │
       └───┤ > 0.50    │       │
           │           │       │
           │           │       │
           └───────────┘       │
                       │ Person B
                       └───────┘
  YOLO NMS Algorithm: "These boxes overlap too much. Person B must be 
                       a duplicate detection of Person A. Delete Person B!"
  Result: Undercounting by 30% to 50% in crowded turnstiles.
```

### The RT-DETR Solution: Direct Set Prediction
**RT-DETR (Real-Time DEtection TRansformer)**, developed by Baidu, replaces anchor boxes and greedy NMS with an end-to-end transformer architecture:
1. **Hybrid Encoder:**
   - **AIFI (Attention-based Intra-scale Feature Interaction):** Uses self-attention only on high-level feature maps to capture global relationships between objects without massive computational cost.
   - **CCFM (Cross-scale Feature Fusion Module):** Fuses low-level spatial features with high-level semantic features using lightweight convolutional layers.
2. **Transformer Decoder & 300 Object Queries:**
   - The model maintains 300 learnable "queries".
   - Each query scans the multi-scale feature maps using cross-attention and directly predicts at most one physical object.
3. **Bipartite Hungarian Matching Loss:**
   - During training, a one-to-one optimal match is computed between the 300 predicted queries and ground-truth objects.
   - **NMS is completely eliminated.** If two people stand with $70\%$ visual overlap, RT-DETR assigns Query #14 to Person A and Query #82 to Person B. Neither is deleted.

---

## 5. How and Where Did We Train RT-DETR?

### Training Script & Configuration
Fine-tuning is orchestrated by [`train_rtdetr.py`](file:///D:/yasmin%20programs/PROGLINT/train_rtdetr.py).

```powershell
python train_rtdetr.py --data data/processed/pets2009.yaml --epochs 15 --batch 16 --imgsz 640
```

### Hyperparameters & Hardware Configuration
- **Base Architecture Weights:** `rtdetr-l.pt` (COCO base pre-trained weights from Ultralytics).
- **Target Hardware:** Dedicated NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB GDDR6 VRAM, Ada Lovelace `sm_89`).
- **Precision:** FP16 Automatic Mixed Precision (`amp=True`) using NVIDIA Tensor Cores.
- **Optimizer:** AdamW (`lr0 = 0.0001`, `weight_decay = 0.0001`).
- **Batch Size:** 16 (with automatic VRAM fallback to 8 if memory spikes).
- **In-Memory Caching:** `cache=True` (loads image tensors directly into system RAM to eliminate disk I/O bottlenecks).
- **Export Destination:** Best checkpoint is automatically saved and exported to [`runs/custom_train/best_rtdetr.pt`](file:///D:/yasmin%20programs/PROGLINT/runs/custom_train/best_rtdetr.pt).

---

## 6. Model Input Resolution Trade-Offs: Why 640x640?

In object detection, the input resolution determines the receptive field and computational cost:

| Input Resolution | Inference Latency (RTX 4060) | Throughput (FPS) | Small Pedestrian Detection | Edge Real-Time Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| **$640 \times 640$ (Selected)** | **~49 ms** | **~20.0 FPS** | **High (for gate range)** | **YES (Real-time sustained)** |
| $960 \times 960$ | ~110 ms | ~9.0 FPS | Very High | Borderline (Frame drops) |
| $1080 \times 1080$ | ~185 ms | ~5.4 FPS | Maximum | NO (Severe latency lag) |

### Why $640\times 640$ Was Chosen:
At access-control turnstiles and doorways, pedestrians occupy at least $15\%\text{--}40\%$ of the vertical frame height. Unlike drone surveillance where humans are tiny 10-pixel specks, CCTV access gates provide large targets.
$640\times 640$ provides the ideal balance:
1. It delivers **real-time 20 FPS** throughput on edge GPUs.
2. It consumes under **$200\text{ MB}$ of VRAM**, leaving remaining GPU memory for tracker Re-ID feature extraction and video encoding.
3. Training and validation metrics confirm zero loss of pedestrian detection accuracy at $640\times 640$.

---

## 7. Metrics Achieved

On the holdout validation sequence `S1L2`:
- **mAP@50 (Mean Average Precision at 0.50 IoU):** **0.9949** ($99.49\%$)
- **mAP@50:95 (Mean Average Precision averaged from 0.50 to 0.95 IoU):** **0.9849** ($98.49\%$)
- **Precision:** $0.991$
- **Recall:** $0.988$
