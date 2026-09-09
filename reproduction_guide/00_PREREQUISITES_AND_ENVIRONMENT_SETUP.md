# Chapter 0: Prerequisites & Environment Setup (From Absolute Scratch)

Welcome to the **From-Scratch Reproduction Guide** for ProGlint Footfall Intelligence.
In this chapter, you will set up the entire development environment on a clean Windows machine with zero prior dependencies.

---

## 1. What Technologies Are We Using and Why?

Before installing anything, understand what each component does:
1. **Python 3.10+:** The core programming language.
2. **PyTorch with CUDA:** The deep learning framework that runs tensor operations on your NVIDIA GPU instead of the slow CPU.
3. **Ultralytics:** The enterprise computer vision library containing the RT-DETR model and BoT-SORT tracker.
4. **OpenCV (`opencv-python`):** The image and video manipulation library used to decode MP4 videos, resize frames, and draw bounding boxes and tripwires.
5. **FastAPI & Uvicorn:** The backend web framework used to expose our computer vision engine as an asynchronous HTTP REST API.
6. **Streamlit:** The interactive Python frontend used to build the web dashboard and calibration sliders.
7. **FFmpeg:** The multimedia transcoding engine that converts raw OpenCV video output into browser-playable H.264 video.

---

## 2. Step 1: Create the Project Directory

Open **PowerShell** on your machine and create a dedicated project folder:

```powershell
mkdir "D:\yasmin programs\PROGLINT"
cd "D:\yasmin programs\PROGLINT"
```

---

## 3. Step 2: Create an Isolated Python Virtual Environment

Never install packages directly into your global Windows Python environment. If packages have conflicting versions, your system will break. We create an isolated sandbox called `cv_env`:

```powershell
python -m venv cv_env
```
What this does:
- Creates a folder called `cv_env/`.
- Copies a clean Python interpreter and `pip` package manager inside it.

To activate the virtual environment:
```powershell
.\cv_env\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` and try again).*

---

## 4. Step 3: Install PyTorch with CUDA Acceleration

To enable your NVIDIA GPU (e.g. RTX 4060), you must install the CUDA-enabled build of PyTorch. Run:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Verification (The 5-Second Test):
Run this command to verify PyTorch can communicate with your GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
You should see:
```text
CUDA Available: True | Device: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 5. Step 4: Install Project Dependencies

Create a `requirements.txt` file with the following packages:

```text
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
streamlit>=1.28.0
plotly>=5.15.0
requests>=2.31.0
pillow>=9.5.0
pyyaml>=6.0
```

Install them all in one command:
```powershell
pip install -r requirements.txt
```

---

## 6. Step 5: Install FFmpeg for Browser Video Playback

OpenCV cannot write web-native H.264 videos on Windows by default. We need FFmpeg to transcode processed videos so web browsers can play them.

1. Download FFmpeg essentials build from [gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/).
2. Extract the archive (e.g., to `C:\ffmpeg`).
3. Add `C:\ffmpeg\bin` to your Windows System `PATH`.
4. Verify by running in PowerShell:
```powershell
ffmpeg -version
```
If you see the FFmpeg copyright and version banner, you are completely set up!

---

## 7. Folder Skeleton to Create

Now create the directory structure for your codebase:

```powershell
mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\scripts
mkdir engine
mkdir backend
mkdir frontend
mkdir evaluation
mkdir runs
mkdir runs\custom_train
```

In the next chapter, we will build **Chapter 1: Data Ingestion & Annotation Parsing from Scratch**.
