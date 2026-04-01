"""
analysis.py
-----------
Optional post-processing script.

Features:
  1. Person-count-over-time plot  (from count_over_time.csv)
  2. Speed estimation per track   (pixels / frame → annotated CSV)
  3. Top-view / bird's-eye projection  (if homography matrix is provided)

Usage:
    python analysis.py
    python analysis.py --csv ../data/output/count_over_time.csv
"""

import argparse
import csv
import os

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠️  matplotlib not installed – skipping plot generation")


# ── 1. Count-over-time plot ──────────────────────────────────────────────────

def plot_count_over_time(csv_path: str, out_path: str):
    frames, counts = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            counts.append(int(row["person_count"]))

    if not HAS_MPL:
        print("matplotlib missing – skipping plot")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(frames, counts, linewidth=1.2, color="#2196F3")
    plt.fill_between(frames, counts, alpha=0.15, color="#2196F3")
    plt.xlabel("Frame number")
    plt.ylabel("People detected")
    plt.title("Person Count Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📈  Count plot saved → {out_path}")


# ── 2. Speed estimation ──────────────────────────────────────────────────────
# (Works with the trajectory history exported from visualize._history)

def estimate_speeds(history: dict, fps: int, pixels_per_metre: float = 10.0) -> dict:
    """
    Compute average speed (m/s) for each track ID.

    Args:
        history:           {track_id: [(cx, cy), …]}
        fps:               Video frame rate.
        pixels_per_metre:  Calibration factor (pixels per metre on the pitch).

    Returns:
        {track_id: average_speed_mps}
    """
    speeds = {}
    for tid, pts in history.items():
        pts = list(pts)
        if len(pts) < 2:
            speeds[tid] = 0.0
            continue
        dists = [
            np.linalg.norm(np.array(pts[i]) - np.array(pts[i - 1]))
            for i in range(1, len(pts))
        ]
        total_dist_m = sum(dists) / pixels_per_metre
        total_time_s = len(dists) / fps
        speeds[tid] = total_dist_m / total_time_s if total_time_s > 0 else 0.0
    return speeds


# ── 3. Bird's-eye projection ─────────────────────────────────────────────────

def birds_eye_frame(
    frame: np.ndarray,
    tracks: list,
    H: np.ndarray,
    out_size: tuple[int, int] = (400, 600),
) -> np.ndarray:
    """
    Project tracked foot positions onto a top-down view using homography H.

    Args:
        frame:    Original BGR frame (unused directly, for reference).
        tracks:   List of (x1, y1, x2, y2, track_id).
        H:        3×3 homography matrix from cv2.findHomography.
        out_size: (height, width) of the output canvas in pixels.

    Returns:
        BGR image of the bird's-eye view with coloured dots.
    """
    from visualize import get_color

    canvas = np.zeros((out_size[0], out_size[1], 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)   # dark grey background

    for x1, y1, x2, y2, tid in tracks:
        fx = (x1 + x2) / 2.0
        fy = float(y2)           # foot point

        # Apply homography
        pt = np.array([[[fx, fy]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, H)
        px, py = int(dst[0][0][0]), int(dst[0][0][1])

        if 0 <= px < out_size[1] and 0 <= py < out_size[0]:
            color = get_color(tid)
            cv2.circle(canvas, (px, py), 6, color, -1)
            cv2.putText(canvas, str(tid), (px + 7, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return canvas


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Post-processing analysis tools")
    p.add_argument("--csv",  default="../data/output/count_over_time.csv")
    p.add_argument("--plot", default="../data/output/count_plot.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.csv):
        plot_count_over_time(args.csv, args.plot)
    else:
        print(f"CSV not found: {args.csv}")
        print("Run main.py first to generate tracking output.")