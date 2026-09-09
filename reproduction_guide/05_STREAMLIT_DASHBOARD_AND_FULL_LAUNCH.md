# Chapter 5: Streamlit Dashboard — The 15-Line Truth

In [`frontend/app.py`](file:///D:/yasmin%20programs/PROGLINT/frontend/app.py), you see Plotly graphs, layout columns, session states, and image conversions.

Strip away the visual polish, and **the web frontend is literally 15 lines of Python**.

---

## 1. The Pure 15-Line Frontend Code

```python
import streamlit as st
import requests

st.title("ProGlint Footfall Intelligence")

# 1. Upload video file widget
video = st.file_uploader("Upload Surveillance Video", type=["mp4"])

if video and st.button("Start Counting"):
    # 2. Send video file to our FastAPI backend (Chapter 4)
    files = {"video": (video.name, video.read(), "video/mp4")}
    res = requests.post("http://127.0.0.1:8000/process_video", files=files).json()

    # 3. Show live scoreboard
    m = res["metrics"]
    st.metric("Total IN", m["total_in"])
    st.metric("Total OUT", m["total_out"])
    st.metric("Net Occupancy", m["current_occupancy"])
```

**That is the whole frontend.**

---

## 2. Why Did `frontend/app.py` Have 90 Lines?

The other 75 lines were added for user convenience:
1. **Calibration Sliders (Line A & Line B):** So users can visually adjust where the lines are drawn instead of hardcoding $0.45$ and $0.55$.
2. **First-Frame Preview:** Extracts frame 0 and shows Line A (Cyan) and Line B (Magenta) overlaid so the user can verify alignment.
3. **Plotly Graph:** Draws a line chart showing how occupancy increased and decreased over time.

---

## 3. The 30-Second Summary of the Entire Project

Now you can see the entire project with crystal clarity:

```
┌──────────────────────────────────────┬────────────────────────────────────────┐
│ Module                               │ The Actual Core Logic                  │
├──────────────────────────────────────┼────────────────────────────────────────┤
│ 1. Data Conversion (Chapter 1)       │ 15 lines: Parse XML, divide by W & H,  │
│                                      │           write .txt label file.       │
│                                      │                                        │
│ 2. RT-DETR Training (Chapter 2)      │ 5 lines:  model = RTDETR('rtdetr-l.pt')│
│                                      │           model.train(...)             │
│                                      │                                        │
│ 3. Tracking & Counting (Chapter 3)   │ 25 lines: model.track(), check feet cy,│
│                                      │           if A->B then IN, if B->A OUT.│
│                                      │                                        │
│ 4. FastAPI Backend (Chapter 4)       │ 15 lines: @app.post, run engine,       │
│                                      │           return JSON.                 │
│                                      │                                        │
│ 5. Streamlit Frontend (Chapter 5)    │ 15 lines: upload video, post to API,   │
│                                      │           show metrics on screen.      │
└──────────────────────────────────────┴────────────────────────────────────────┘
```

The entire real intelligence of this project is **less than 80 lines of Python code**. 

Everything else in the repository is just professional packaging, error handling, and visual formatting. When you are learning or presenting, **keep these 80 lines in mind**, and you will never feel lost or confused again!
