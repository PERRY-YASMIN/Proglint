"""run.py - Unified Orchestration Launcher for ProGlint Services.

What is where:
- run_backend(): Starts Uvicorn FastAPI backend on port 8000.
- run_frontend(): Starts Streamlit frontend dashboard on port 8501.
- kill_proc_tree(): Safely terminates child process trees on Windows and Linux.
- main(): Parses CLI mode ('backend', 'frontend', or 'both') and manages lifecycle.
"""

# ==============================================================================
# 1. IMPORTS & SUBPROCESS TOOLS
# ==============================================================================
# [FROM: Python Standard Library] subprocess for launching child processes, sys for python binary
import subprocess, sys, time, os

# [DEF: run_backend()] Launches the FastAPI backend microservice via Uvicorn
# [FUNCTION: Starts Uvicorn on 0.0.0.0:8000 with backend/main.py]
# [USED IN: main() when mode is 'backend' or 'both']
def run_backend():
    print("[Launcher] Starting FastAPI backend on http://127.0.0.1:8000...")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])

# [DEF: run_frontend()] Launches the Streamlit web analytics dashboard
# [FUNCTION: Starts Streamlit on port 8501 targeting frontend/app.py]
# [USED IN: main() when mode is 'frontend' or 'both']
def run_frontend():
    print("[Launcher] Starting Streamlit frontend on http://127.0.0.1:8501...")
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"])

# [DEF: kill_proc_tree()] Gracefully terminates a process and all its child sub-processes
# [FUNCTION: Cross-Platform Process Kill] Uses Windows 'taskkill /F /T' or Linux 'proc.terminate()'
# [USED IN: Ctrl+C signal handler inside main()]
def kill_proc_tree(proc):
    if proc and proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()

# [DEF: main()] Main CLI orchestrator managing system startup and clean exit
# [USED IN: Terminal command: python run.py both]
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
        # Launch backend first, wait 2 seconds for Uvicorn to bind port, then launch frontend
        b_proc = run_backend()
        time.sleep(2)
        f_proc = run_frontend()
        try:
            while True: time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[Launcher] Terminating services...")
            kill_proc_tree(b_proc)
            kill_proc_tree(f_proc)
            print("[Launcher] Ports 8000 and 8501 are clean.")
    else:
        print(f"Unknown mode '{mode}'. Usage: python run.py [backend|frontend|both]")

if __name__ == "__main__":
    main()
