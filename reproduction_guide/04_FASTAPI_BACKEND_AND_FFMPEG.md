# Chapter 4: FastAPI Backend — The 15-Line Truth

In [`backend/main.py`](file:///D:/yasmin%20programs/PROGLINT/backend/main.py), you see CORS middlewares, UUID string generators, exception handlers, and subprocess calls.

Strip that away, and **the backend API is literally 15 lines of Python**.

---

## 1. The Pure 15-Line Backend API

```python
import os
from fastapi import FastAPI, UploadFile, File
from engine.tracker import FootfallEngine

# 1. Initialize API app and FootfallEngine
app = FastAPI()
engine = FootfallEngine()

# 2. Define the video processing endpoint
@app.post("/process_video")
async def process_video(video: UploadFile = File(...)):
    # Save the uploaded video bytes to a temporary file
    temp_path = "temp_input.mp4"
    with open(temp_path, "wb") as f:
        f.write(await video.read())

    # Run the 25-line tracking engine from Chapter 3
    results = engine.process_video(temp_path, "runs/output.mp4")

    # Return the metrics directly to whoever called the API
    return results
```

**That is the whole backend API.**

---

## 2. Why Did `backend/main.py` Have 70 Lines?

The other 55 lines were just two helpers:
1. **CORS Middleware (5 lines):** Allows web browsers to connect without security blocks (`allow_origins=["*"]`).
2. **FFmpeg Transcoding (10 lines):** OpenCV writes raw videos with a codec (`mp4v`) that modern web browsers (Chrome, Edge) refuse to play. Running `ffmpeg -i output.mp4 -vcodec libx264 browser_output.mp4` converts it to standard H.264 so it can stream in the browser.

**What you need to remember:**  
A backend API is simply a wrapper function: **Receive uploaded video bytes $\rightarrow$ Pass to `FootfallEngine` $\rightarrow$ Return JSON counts**.
