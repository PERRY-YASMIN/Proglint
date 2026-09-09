# Chapter 3: Tracking & Counting — The 25-Line Truth (Person 3 Core)

In [`engine/tracker.py`](file:///D:/yasmin%20programs/PROGLINT/engine/tracker.py), there are classes, methods, kwargs, and dozens of OpenCV drawing functions.

When you strip away the user interface and video-writing boilerplate, **the entire tracking and counting engine is literally 25 lines of Python**.

---

## 1. The Pure 25-Line Counting Engine

Here is the complete core of Person 3's accounting engine:

```python
import cv2
from ultralytics import RTDETR

# 1. Load model and open video
model = RTDETR("runs/custom_train/best_rtdetr.pt")
cap = cv2.VideoCapture("TownCentre_test.mp4")

total_in, total_out = 0, 0
history = {}  # Stores: tid -> (last_y, state)
y_a, y_b = 486, 594  # Line A (45% height) and Line B (55% height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 2. Detect & Track with BoT-SORT (Returns boxes with persistent IDs)
    res = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], conf=0.40, verbose=False)
    if not res[0].boxes or res[0].boxes.id is None: continue

    boxes = res[0].boxes.xyxy.cpu().numpy()
    ids   = res[0].boxes.id.int().cpu().tolist()

    # 3. For each person, evaluate foot point and FSM crossing
    for box, tid in zip(boxes, ids):
        cy = int(box[3])  # box[3] is y2: bottom of box (FEET contact with floor!)
        last_y, state = history.get(tid, (cy, "IDLE"))

        # The 4-State Machine:
        if state == "IDLE":
            if last_y < y_a <= cy:    state = "PENDING_IN"   # Crossed Line A downwards
            elif last_y > y_b >= cy:  state = "PENDING_OUT"  # Crossed Line B upwards
        elif state == "PENDING_IN" and last_y < y_b <= cy:
            total_in += 1
            state = "DONE"  # State lock: Never count this ID again!
        elif state == "PENDING_OUT" and last_y > y_a >= cy:
            total_out += 1
            state = "DONE"  # State lock: Never count this ID again!

        history[tid] = (cy, state)

# 4. Final Analytics
print(f"Scoreboard -> IN: {total_in} | OUT: {total_out} | OCCUPANCY: {total_in - total_out}")
```

**That is the whole counting engine.**

---

## 2. Why Does This 25-Line Code Work So Reliably?

Look at the 3 genius decisions in these 25 lines:

1. **`cy = int(box[3])` (The Foot Point):**
   `box[3]` is $y_2$ (the bottom edge of the box). In overhead CCTV, the center of the body moves when a person leans or swings their arms. But shoes touch the floor rigidly. Using `box[3]` gives a smooth, jitter-free coordinate.
2. **`y_a < cy <= y_b` (The Two-Line Deadband):**
   By requiring a person to cross Line A *and then* cross Line B, loitering or pacing back and forth never triggers a count.
3. **`state = "DONE"` (The Duplicate Prevention Lock):**
   Once `total_in += 1` is triggered, the state becomes `DONE`. Notice that transitions to `PENDING` only happen if `state == "IDLE"`. As long as that person stays in the frame, they can never trigger a second count!

---

## 3. What Was the Other Code in `tracker.py`?

The remaining lines in `tracker.py` were simply:
- **OpenCV Visuals:** `cv2.rectangle` to draw green boxes, `cv2.line` to draw cyan and magenta lines, and `cv2.putText` to print `"IN: 2 | OUT: 1"` in the corner of the video.
- **Garbage Collection:** Deleting tracks from the `history` dictionary if they haven't been seen for 60 frames so memory doesn't grow indefinitely.
- **Timeout:** Resetting `PENDING` back to `IDLE` if someone enters the deadband and stands frozen for $>3$ seconds.

When learning or presenting, **focus on the 25-line loop above**. That is the pure intellectual core of Person 3's role.
