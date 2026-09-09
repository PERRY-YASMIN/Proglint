# Document 6: Inside RT-DETR — Complete Mechanics from Input to Output

## 1. Demystifying the Black Box

When people say *"we use a pre-trained model"*, examiners and interviewers immediately ask:
> *"Do you actually know what happens mathematically inside the neural network from the moment an image enters to the moment bounding boxes exit?"*

This document breaks down the entire internal pipeline of **RT-DETR (Real-Time DEtection TRansformer)** step-by-step, with zero hand-waving.

---

## 2. Complete Architecture Overview

RT-DETR is divided into **4 core stages**:

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: IMAGE PRE-PROCESSING                                               │
 │ OpenCV BGR Frame [1080, 1920, 3] ──► RGB ──► Resize ──► Normalize [0, 1]   │
 │ Tensor Shape: [1, 3, 640, 640]                                              │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: BACKBONE FEATURE EXTRACTION                                        │
 │ Convolutional Backbone (HGNetv2 / ResNet) extracts multi-scale features:    │
 │ • Scale 3 (S3): 80x80 pixels (stride 8)   - Low-level textures, edges       │
 │ • Scale 4 (S4): 40x40 pixels (stride 16)  - Mid-level parts (torso, head)   │
 │ • Scale 5 (S5): 20x20 pixels (stride 32)  - High-level semantic context     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: EFFICIENT HYBRID ENCODER (AIFI + CCFM)                             │
 │ • AIFI: Self-Attention applied ONLY to S5 (20x20 = 400 tokens)              │
 │ • CCFM: Cross-scale Feature Fusion fuses S3, S4, and S5 with convolutions   │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4: TRANSFORMER DECODER & 300 OBJECT QUERIES                           │
 │ • 300 learnable queries act as "spatial detectives"                         │
 │ • Query Self-Attention: Queries communicate to prevent duplicate claims     │
 │ • Cross-Attention: Queries probe the image features for person patterns     │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 5: PREDICTION HEADS & OUTPUT FILTERING                                │
 │ • Class Head: 300 sigmoid confidence scores P(person)                       │
 │ • Box Head: 300 normalized coordinates [xc, yc, w, h]                       │
 │ • Filter: Retain queries with conf >= 0.40                                  │
 │ • Convert to pixels: [x1, y1, x2, y2]                                       │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step 1: Pre-Processing (From Video Frame to Tensor)

Before a pixel touches a neural weight, it undergoes strict tensor formatting:

1. **BGR to RGB:** OpenCV decodes frames in Blue-Green-Red order. RT-DETR was trained on RGB images. Channels are swapped (`cv2.COLOR_BGR2RGB`).
2. **Letterbox Resizing to $640\times 640$:** The image (e.g., $1920\times 1080$) is resized while maintaining aspect ratio, padding the borders with gray pixels ($114, 114, 114$).
3. **Normalization ($[0, 255] \rightarrow [0.0, 1.0]$):** Every pixel integer $p \in [0, 255]$ is divided by $255.0$.
4. **Dimension Permutation & Batching:**
   - OpenCV shape: `(Height=640, Width=640, Channels=3)`
   - PyTorch shape: `(Batch=1, Channels=3, Height=640, Width=640)`
   - Memory representation: A 32-bit floating point tensor $\mathbf{X} \in \mathbb{R}^{1 \times 3 \times 640 \times 640}$ transferred to GPU memory (`cuda:0`).

---

## 4. Step 2: Backbone Feature Extraction (From Pixels to Features)

The input image $\mathbf{X}$ passes through a convolutional backbone (e.g., HGNetv2 or ResNet). 
As the image flows through consecutive convolutional blocks, spatial resolution decreases while channel depth (semantic meaning) increases:

```
  Input Tensor: [3, 640, 640]
       │
       ▼ Conv Stride 8
  Feature Map S3: [256 channels, 80 x 80]  <-- High resolution, low semantics (edges, textures)
       │
       ▼ Conv Stride 16
  Feature Map S4: [512 channels, 40 x 40]  <-- Medium resolution (body parts, coats)
       │
       ▼ Conv Stride 32
  Feature Map S5: [1024 channels, 20 x 20] <-- Low resolution, high semantics (full person context)
```

---

## 5. Step 3: The Hybrid Encoder (AIFI + CCFM)

In original vision transformers (like the original 2020 DETR by Facebook), self-attention was applied across all pixels of all feature maps.
- A $640\times 640$ image generates thousands of spatial tokens.
- Standard self-attention has quadratic complexity $\mathcal{O}(N^2)$. Running self-attention on $80\times 80 = 6400$ tokens requires $(6400)^2 \approx 41\text{ million}$ operations per attention head, destroying real-time speeds (dropping FPS to < 3 FPS).

**RT-DETR's Breakthrough: The Hybrid Encoder**
Baidu discovered that low-level features ($S_3, S_4$) do not need self-attention—they only contain local texture details. Only high-level features ($S_5$) require global semantic reasoning.

```
                  THE HYBRID ENCODER ARCHITECTURE
                  
  Feature Map S5 [20x20 = 400 tokens]
       │
       ▼
  ┌────────────────────────────────────────────────────────┐
  │ AIFI (Attention-based Intra-scale Feature Interaction) │
  │ Runs Multi-Head Self-Attention ONLY on S5.             │
  │ 400 tokens -> (400)^2 = 160,000 ops (Fast & Real-Time!)│
  └────────────────────────┬───────────────────────────────┘
                           │
                           ▼ High-level global context
  ┌────────────────────────────────────────────────────────┐
  │ CCFM (Cross-scale Feature-fusion Module)               │
  │ Uses lightweight convolutions (RepBlocks) to inject    │
  │ the global context from S5 down into S4 and S3.        │
  └────────────────────────┬───────────────────────────────┘
                           │
                           ▼
  Enriched Multi-Scale Features: E3 (80x80), E4 (40x40), E5 (20x20)
```

---

## 6. Step 4: The Transformer Decoder & 300 Object Queries

This is the heart of RT-DETR and the reason it does not need YOLO's Non-Maximum Suppression (NMS).

### What Is an "Object Query"?
Think of the **300 Object Queries** as **300 virtual detectives** assigned to inspect the image.
- Each query $\mathbf{q}_i \in \mathbb{R}^{256}$ is a learnable vector representing a hypothesis: *"I am responsible for detecting an object in a specific region of the image."*
- At the start of inference, RT-DETR uses **IoU-aware Query Selection** to initialize these 300 queries from the most promising feature locations.

### The Two Critical Attention Mechanisms Inside the Decoder:

```
  For each of the 6 Decoder Layers:
  
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. QUERY SELF-ATTENTION (The Detectives Talk to Each Other)                 │
  │ Each of the 300 queries compares its assigned target with all 299 others:   │
  │ Query 14: "I am detecting a person at (x=380, y=210)."                      │
  │ Query 82: "I was looking there too, but Query 14 is a better fit.           │
  │            I will look elsewhere or predict background."                    │
  │ ──► THIS ELIMINATES DUPLICATES WITHOUT NEEDING NMS!                         │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 2. CROSS-ATTENTION (The Detectives Inspect the Image)                       │
  │ Each query compares its representation against the encoded image features:  │
  │ Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d) ) * V                     │
  │ Where:                                                                      │
  │ • Q (Queries): The 300 detective vectors.                                   │
  │ • K (Keys): Multi-scale image features from the Hybrid Encoder.             │
  │ • V (Values): The actual visual image representations.                      │
  │ ──► Extracts precise pedestrian boundaries and clothing contours.           │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Step 5: Prediction Heads & Coordinate Projection

At the end of the transformer decoder, all 300 queries are fed into two parallel Multi-Layer Perceptrons (MLPs):

### 1. Classification Head (Linear Projection $\rightarrow$ Sigmoid):
- Computes class probability $P(\text{person})$ for each of the 300 queries.
- Outputs a vector $\mathbf{c} \in [0.0, 1.0]^{300}$.

### 2. Box Regression Head (3-Layer MLP with ReLU):
- Predicts normalized bounding box geometry for each query:
$$\mathbf{b}_i = [x_c, y_c, w, h] \in [0.0, 1.0]^4$$

---

## 8. Step 6: Post-Processing Filter (From Tensor to Engine)

In [`engine/tracker.py`](file:///D:/yasmin%20programs/PROGLINT/engine/tracker.py#L42-L56), we receive the 300 query predictions:

```
300 Raw Query Proposals:
Query 1:   conf = 0.01  [xc=0.12, yc=0.15, w=0.04, h=0.08]  --> REJECTED (< 0.40)
Query 2:   conf = 0.03  [xc=0.88, yc=0.92, w=0.05, h=0.10]  --> REJECTED (< 0.40)
...
Query 14:  conf = 0.89  [xc=0.45, yc=0.36, w=0.07, h=0.22]  --> ACCEPTED (Person A!)
Query 82:  conf = 0.84  [xc=0.52, yc=0.48, w=0.08, h=0.24]  --> ACCEPTED (Person B!)
```

### Unscaling to Physical Frame Dimensions:
For each accepted query, coordinates are multiplied by video frame width $W = 1920$ and height $H = 1080$:
$$x_1 = \left( x_c - \frac{w}{2} \right) \times W = \left( 0.45 - \frac{0.07}{2} \right) \times 1920 = 796.8\text{ px}$$
$$y_1 = \left( y_c - \frac{h}{2} \right) \times H = \left( 0.36 - \frac{0.22}{2} \right) \times 1080 = 270.0\text{ px}$$
$$x_2 = \left( x_c + \frac{w}{2} \right) \times W = \left( 0.45 + \frac{0.07}{2} \right) \times 1920 = 931.2\text{ px}$$
$$y_2 = \left( y_c + \frac{h}{2} \right) \times H = \left( 0.36 + \frac{0.22}{2} \right) \times 1080 = 507.6\text{ px}$$

The box $[797, 270, 931, 508]$ is passed directly to Person 2's **BoT-SORT** tracker!

---

## 9. Quick Summary for Your Defense

If asked: *"What happens inside RT-DETR?"*

> *"RT-DETR takes an input frame, resizes it to $640\times 640$, and extracts multi-scale features ($S_3, S_4, S_5$) using a convolutional backbone.  
> It then passes them to an Efficient Hybrid Encoder: AIFI runs multi-head self-attention on the high-level $S_5$ map to capture global scene context, while CCFM fuses these features across scales using lightweight convolutions.  
> The fused features enter a Transformer Decoder with 300 learnable object queries. Through query self-attention, the queries communicate with each other to avoid claiming the same pedestrian, eliminating the need for NMS. Through cross-attention, each query extracts bounding box coordinates and class confidence scores from the image features.  
> Finally, predictions with confidence $\ge 0.40$ are unscaled to original video resolution and passed to the BoT-SORT tracker."*
