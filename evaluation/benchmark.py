"""evaluation/benchmark.py - Benchmark Latency and Footfall Accuracy on Video.

What is where:
- run_benchmark(): Runs FootfallEngine on a video clip, measures per-frame latency,
                   calculates average FPS, and writes BENCHMARK_REPORT.md.
- CLI entrypoint: Parses --video, --max_frames, --output_report arguments.
"""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================
# [FROM: Python Standard Library] argparse for CLI options, os/sys for filesystem and module paths, time for high-resolution latency profiling
import argparse, os, sys, time
# [FROM: opencv-python] cv2 for decoding video frames via VideoCapture
import cv2

# [INIT: Repository Root Path] Adds parent folder to sys.path so 'engine' can be imported
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path: sys.path.insert(0, _REPO)

# [FROM: engine.tracker] Imports FootfallEngine (RT-DETR + BoT-SORT + Tripwire FSM)
from engine.tracker import FootfallEngine

# ==============================================================================
# 2. BENCHMARK EXECUTION LOGIC
# ==============================================================================
# [DEF: run_benchmark()] Measures latency (ms/frame), throughput (FPS), and count metrics
# [USED IN: CLI execution below, or automated evaluation pipelines]
def run_benchmark(video_path: str = "TownCentre_test.mp4", max_frames: int = 140,
                  output_report: str = "evaluation/BENCHMARK_REPORT.md") -> str:
    """Benchmark FootfallEngine latency, FPS, and counts on a video clip."""
    print(f"[Benchmark] Evaluating '{video_path}' (up to {max_frames} frames)...")

    # [INIT: FootfallEngine] Initializes detector (RT-DETR) and tracker (BoT-SORT)
    # [FROM: engine/tracker.py]
    engine = FootfallEngine()

    # [INIT: VideoCapture] Opens video source file via OpenCV
    # [FROM: cv2.VideoCapture]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {video_path}")

    # [INIT: State & Timing Variables] Track per-frame latency list and counter
    latencies, frame_idx = [], 0
    t_start = time.perf_counter()

    # [FUNCTION: Frame-by-Frame Profiling Loop]
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None: break

        # [FUNCTION: Latency Measurement] Time only the engine inference and tracking logic
        t0 = time.perf_counter()
        engine.process_frame(frame, frame_idx=frame_idx)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        frame_idx += 1

    # [FUNCTION: Cleanup] Release video capture handle
    cap.release()

    # [FUNCTION: Summary Metric Computations]
    total_time = time.perf_counter() - t_start
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0

    # [DEF: Markdown Report] Formatted markdown table of latency, FPS, and footfall counts
    # [USED IN: Written to output_report file]
    report = f"""# ProGlint Footfall Intelligence Benchmark Report

| Metric | Measured Value |
| :--- | :--- |
| **Model** | `runs/custom_train/best_rtdetr.pt` (RT-DETR-R18) |
| **Tracker** | `botsort.yaml` |
| **Device** | `{engine.device}` |
| **Processed Frames** | {frame_idx} |
| **Mean Pipeline Latency** | {avg_latency:.2f} ms |
| **Sustained FPS** | {avg_fps:.2f} FPS |
| **Total IN** | {engine.total_in} |
| **Total OUT** | {engine.total_out} |
| **Current Occupancy** | {engine.occupancy} |
| **Unique Track IDs** | {len(engine.seen_ids)} |
"""
    # [FUNCTION: Save Report to Disk]
    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Benchmark] Saved report to: {output_report}")
    return report

# ==============================================================================
# 3. CLI ENTRYPOINT
# ==============================================================================
# [FUNCTION: Main CLI Driver] Allows running benchmark directly from terminal
# [USED IN: python evaluation/benchmark.py --video TownCentre_test.mp4 --max_frames 140]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProGlint Benchmark")
    parser.add_argument("--video", type=str, default="TownCentre_test.mp4")
    parser.add_argument("--max_frames", type=int, default=140)
    parser.add_argument("--output_report", type=str, default="evaluation/BENCHMARK_REPORT.md")
    args = parser.parse_args()
    print("\n" + run_benchmark(args.video, args.max_frames, args.output_report))
