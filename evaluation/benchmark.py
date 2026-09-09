"""evaluation/benchmark.py - Streamlined Benchmark Suite for ProGlint Footfall Intelligence."""
import argparse, os, sys, time, cv2

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path: sys.path.insert(0, _REPO)

from engine.tracker import FootfallEngine

def run_benchmark(video_path="TownCentre_test.mp4", max_frames=140, output_report="evaluation/BENCHMARK_REPORT.md"):
    print(f"[Benchmark] Evaluating '{video_path}' (up to {max_frames} frames)...")
    engine = FootfallEngine()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {video_path}")

    latencies, frame_idx = [], 0
    t_start = time.perf_counter()

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None: break
        t0 = time.perf_counter()
        engine.process_frame(frame, frame_idx=frame_idx)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        frame_idx += 1

    cap.release()
    total_time = time.perf_counter() - t_start
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0

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
    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Benchmark] Saved report to: {output_report}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProGlint Benchmark")
    parser.add_argument("--video", type=str, default="TownCentre_test.mp4")
    parser.add_argument("--max_frames", type=int, default=140)
    parser.add_argument("--output_report", type=str, default="evaluation/BENCHMARK_REPORT.md")
    args = parser.parse_args()

    report = run_benchmark(args.video, args.max_frames, args.output_report)
    print("\n" + report)