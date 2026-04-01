"""
evaluation.py
-------------
Two evaluation tools:

1. SIMPLE METRICS (no ground truth needed)
   - Track fragmentation: how often does a track restart (ID switch proxy)
   - Track longevity: distribution of track lifetimes
   - Detection density: average detections per frame

2. MODEL COMPARISON  (YOLOv8n vs YOLOv8s on a sample clip)
   - Runs both models on the same N frames
   - Reports: inference time, detection count, confidence distribution

Usage:
    python evaluation.py --csv ../data/output/track_data.csv
    python evaluation.py --compare --video ../data/input/video.mp4 --frames 100
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRACKLET METRICS  (from track_data.csv)
# ══════════════════════════════════════════════════════════════════════════════

def load_track_data(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({k: int(v) for k, v in row.items()})
    return rows


def compute_tracklet_metrics(track_data: list[dict]) -> dict:
    """
    Compute proxy metrics that don't require ground-truth annotations.

    Returns dict with:
      - n_unique_ids:       Total unique track IDs assigned.
      - avg_track_lifetime: Mean frames a track stays alive.
      - min/max_lifetime:   Min / max track lifetimes.
      - avg_detections_per_frame: Mean confirmed detections per processed frame.
      - fragmentation_ratio: (n_ids - expected_players) / n_ids
                             Lower → fewer ID switches.
      - track_lifetimes:    List of per-track frame counts.
    """
    by_id: dict[int, list[int]] = defaultdict(list)
    by_frame: dict[int, int]    = defaultdict(int)

    for r in track_data:
        by_id[r["track_id"]].append(r["frame"])
        by_frame[r["frame"]] += 1

    lifetimes = [max(frames) - min(frames) + 1 for frames in by_id.values()]

    n_ids   = len(by_id)
    n_frames = len(by_frame)
    avg_det  = np.mean(list(by_frame.values())) if by_frame else 0

    # Heuristic: true player count ≈ mode of detections per frame
    mode_count = int(np.median(list(by_frame.values()))) if by_frame else 1
    fragmentation = (n_ids - mode_count) / max(n_ids, 1)

    return {
        "n_unique_ids":              n_ids,
        "avg_track_lifetime_frames": round(np.mean(lifetimes), 1) if lifetimes else 0,
        "min_lifetime_frames":       int(np.min(lifetimes)) if lifetimes else 0,
        "max_lifetime_frames":       int(np.max(lifetimes)) if lifetimes else 0,
        "avg_detections_per_frame":  round(float(avg_det), 2),
        "fragmentation_ratio":       round(float(fragmentation), 3),
        "track_lifetimes":           lifetimes,
    }


def print_metrics(m: dict):
    print("\n" + "=" * 50)
    print("  TRACKLET EVALUATION METRICS")
    print("=" * 50)
    print(f"  Unique IDs assigned         : {m['n_unique_ids']}")
    print(f"  Avg track lifetime (frames) : {m['avg_track_lifetime_frames']}")
    print(f"  Min / Max lifetime (frames) : {m['min_lifetime_frames']} / {m['max_lifetime_frames']}")
    print(f"  Avg detections / frame      : {m['avg_detections_per_frame']}")
    print(f"  Fragmentation ratio         : {m['fragmentation_ratio']:.3f}  (lower = better)")
    print("=" * 50)
    print()


def plot_lifetime_histogram(lifetimes: list[int], out_path: str):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Track lifetime distribution
    axes[0].hist(lifetimes, bins=20, color="#4CAF50", edgecolor="white")
    axes[0].set_xlabel("Track lifetime (frames)")
    axes[0].set_ylabel("Number of tracks")
    axes[0].set_title("Track Lifetime Distribution")
    axes[0].grid(alpha=0.3)

    # Sorted track lengths (like MOT-style plot)
    axes[1].bar(range(len(lifetimes)),
                sorted(lifetimes, reverse=True),
                color="#2196F3", width=1.0)
    axes[1].set_xlabel("Track rank (longest first)")
    axes[1].set_ylabel("Duration (frames)")
    axes[1].set_title("Track Duration by Rank")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊  Lifetime plot → {out_path}")


def plot_count_over_time(count_csv: str, out_path: str):
    if not HAS_MPL:
        return
    frames, counts = [], []
    with open(count_csv) as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            counts.append(int(row["person_count"]))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(frames, counts, linewidth=1.2, color="#2196F3")
    ax.fill_between(frames, counts, alpha=0.15, color="#2196F3")

    # Rolling average
    if len(counts) > 20:
        roll = np.convolve(counts, np.ones(20) / 20, mode="valid")
        ax.plot(frames[10:-9], roll, color="#F44336",
                linewidth=2, label="20-frame rolling avg")
        ax.legend()

    ax.set_xlabel("Frame number")
    ax.set_ylabel("People detected")
    ax.set_title("Person Count Over Time")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📈  Count plot    → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL COMPARISON  (YOLOv8n vs YOLOv8s)
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(video_path: str, n_frames: int = 100,
                   out_path: str = "../data/output/model_comparison.png"):
    """
    Run YOLOv8n and YOLOv8s on the same frames, compare:
      - inference time (ms/frame)
      - detections per frame (person class, conf ≥ 0.5)
      - confidence distribution
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    frames = []
    count  = 0
    while count < n_frames:
        ret, f = cap.read()
        if not ret:
            break
        if count % 2 == 0:
            frames.append(f)
        count += 1
    cap.release()

    models_info = [
        ("YOLOv8n", "yolov8n.pt"),
        ("YOLOv8s", "yolov8s.pt"),
    ]

    results_summary = {}

    for name, weights in models_info:
        print(f"  Testing {name} on {len(frames)} frames …")
        try:
            model = YOLO(weights)
        except Exception as e:
            print(f"  Could not load {weights}: {e}")
            continue

        times, det_counts, confs = [], [], []

        for frame in frames:
            t0 = time.perf_counter()
            res = model(frame, conf=0.5, verbose=False)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

            dets = 0
            for r in res:
                if r.boxes:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            dets += 1
                            confs.append(float(box.conf[0]))
            det_counts.append(dets)

        results_summary[name] = {
            "avg_ms":   round(np.mean(times), 1),
            "std_ms":   round(np.std(times),  1),
            "avg_dets": round(np.mean(det_counts), 2),
            "confs":    confs,
        }
        print(f"    {name}: {results_summary[name]['avg_ms']} ms/frame, "
              f"{results_summary[name]['avg_dets']} dets/frame")

    if not results_summary or not HAS_MPL:
        return

    # ── Plot comparison ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names  = list(results_summary.keys())
    colors = ["#2196F3", "#4CAF50"]

    # Inference time
    ax = axes[0]
    means = [results_summary[n]["avg_ms"] for n in names]
    stds  = [results_summary[n]["std_ms"] for n in names]
    bars  = ax.bar(names, means, yerr=stds, color=colors,
                   capsize=6, edgecolor="white")
    ax.set_ylabel("ms / frame")
    ax.set_title("Inference Time")
    ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

    # Detection count
    ax = axes[1]
    dets = [results_summary[n]["avg_dets"] for n in names]
    bars = ax.bar(names, dets, color=colors, edgecolor="white")
    ax.set_ylabel("Avg detections / frame")
    ax.set_title("Detection Count (person, conf≥0.5)")
    ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, dets):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    # Confidence distribution
    ax = axes[2]
    for name, color in zip(names, colors):
        if results_summary[name]["confs"]:
            ax.hist(results_summary[name]["confs"], bins=20, alpha=0.6,
                    color=color, label=name, edgecolor="white")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Detection Confidence Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("YOLOv8n vs YOLOv8s Model Comparison", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊  Model comparison plot → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="../data/output/track_data.csv")
    p.add_argument("--count-csv", default="../data/output/count_over_time.csv")
    p.add_argument("--out-dir", default="../data/output")
    p.add_argument("--compare", action="store_true",
                   help="Run model comparison (downloads yolov8s.pt)")
    p.add_argument("--video",   default="../data/input/video.mp4")
    p.add_argument("--frames",  type=int, default=100,
                   help="Number of frames for model comparison")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.csv):
        data    = load_track_data(args.csv)
        metrics = compute_tracklet_metrics(data)
        print_metrics(metrics)

        plot_lifetime_histogram(
            metrics["track_lifetimes"],
            os.path.join(args.out_dir, "track_lifetime_plot.png")
        )

    if os.path.exists(args.count_csv):
        plot_count_over_time(
            args.count_csv,
            os.path.join(args.out_dir, "count_plot.png")
        )

    if args.compare:
        print("\n🔬  Running model comparison …")
        compare_models(
            args.video, n_frames=args.frames,
            out_path=os.path.join(args.out_dir, "model_comparison.png")
        )
