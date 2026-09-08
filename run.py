"""
Launcher script to run the Footfall Analytics backend and frontend.

Usage:
  python run.py backend    # Run FastAPI service on http://127.0.0.1:8000
  python run.py frontend   # Run Streamlit UI on http://127.0.0.1:8501
  python run.py both       # Run both concurrently
"""

import subprocess
import sys
import time


def run_backend():
    print("[Launcher] Starting FastAPI backend on http://127.0.0.1:8000...")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])


def run_frontend():
    print("[Launcher] Starting Streamlit frontend on http://127.0.0.1:8501...")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"])


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if mode == "backend":
        proc = run_backend()
        proc.wait()
    elif mode == "frontend":
        proc = run_frontend()
        proc.wait()
    elif mode == "both":
        backend_proc = run_backend()
        time.sleep(2)
        frontend_proc = run_frontend()
        try:
            backend_proc.wait()
            frontend_proc.wait()
        except KeyboardInterrupt:
            print("\n[Launcher] Shutting down services...")
            backend_proc.terminate()
            frontend_proc.terminate()
    else:
        print(f"Unknown mode '{mode}'. Usage: python run.py [backend|frontend|both]")


if __name__ == "__main__":
    main()
