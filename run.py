"""run.py - Unified launcher for ProGlint Footfall Intelligence."""
import subprocess
import sys
import time
import os
import signal

def run_backend():
    print("[Launcher] Starting FastAPI backend on http://127.0.0.1:8000...")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    print("[Launcher] Starting Streamlit frontend on http://127.0.0.1:8501...")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"])

def kill_proc_tree(proc):
    """Cleanly terminates process and its child processes on Windows and Linux."""
    if proc and proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if mode == "backend":
        proc = run_backend()
        try: proc.wait()
        except KeyboardInterrupt: kill_proc_tree(proc)

    elif mode == "frontend":
        proc = run_frontend()
        try: proc.wait()
        except KeyboardInterrupt: kill_proc_tree(proc)

    elif mode == "both":
        b_proc = run_backend()
        time.sleep(2)
        f_proc = run_frontend()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[Launcher] Gracefully terminating services...")
            kill_proc_tree(b_proc)
            kill_proc_tree(f_proc)
            print("[Launcher] Ports 8000 and 8501 are clean.")
    else:
        print(f"Unknown mode '{mode}'. Usage: python run.py [backend|frontend|both]")

if __name__ == "__main__":
    main()