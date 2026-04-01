"""
speed_estimation.py
--------------------
Estimates per-track movement speed from track_data.csv.

Without camera calibration we work in pixel-space.
An optional `pixels_per_metre` argument converts to real-world units.

Outputs:
  - speed_stats.csv     (per track: avg speed, max speed, distance)
  - speed_overlay video (annotates each bounding box with live speed)

Usage:
    python speed_estimation.py
    python speed_estimation.py --csv ../data/output/track_data.csv \
                               --video ../data/input/video.mp4 \
                               --output ../data/output/speed_output.mp4 \
                               --ppm 8.5
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np


# ── Core speed calculation ────────────────────────────────────────────────────

def compute_speeds(track_data: list[dict], fps: int,
                   pixels_per_metre: float = None) -> dict[int, dict]:
    """
    Compute per-track speed statistics.

    Args:
        track_data:       List of row dicts from track_data.csv.
        fps:              Video frame rate.
        pixels_per_metre: If provided, converts pixel distances to metres.
                          Typical basketball court: ~8-10 px/m at broadcast zoom.

    Returns:
        {track_id: {"avg_speed": float, "max_speed": float,
                    "total_distance": float, "unit": str}}
    """
    # Group positions by track id, sorted by frame
    positions: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for row in sorted(track_data, key=lambda r: r["frame"]):
        positions[row["track_id"]].append((row["frame"], row["cx"], row["cy"]))

    stats = {}
    for tid, pts in positions.items():
        if len(pts) < 2:
            continue

        frame_speeds = []
        total_dist   = 0.0

        for i in range(1, len(pts)):
            f0, x0, y0 = pts[i - 1]
            f1, x1, y1 = pts[i]
            dt_frames = max(1, f1 - f0)

            dist_px = np.hypot(x1 - x0, y1 - y0)
            # speed in px/frame → px/sec
            speed_px_s = dist_px / dt_frames * fps
            frame_speeds.append(speed_px_s)
            total_dist += dist_px

        unit = "px/s"
        factor = 1.0
        if pixels_per_metre:
            factor = 1.0 / pixels_per_metre
            unit   = "m/s"

        stats[tid] = {
            "avg_speed":      round(np.mean(frame_speeds)  * factor, 2),
            "max_speed":      round(np.max(frame_speeds)   * factor, 2),
            "total_distance": round(total_dist * factor, 2),
            "unit":           unit,
        }

    return stats


def live_speed_for_frame(frame_num: int,
                          positions_by_id: dict[int, list],
                          fps: int,
                          window: int = 10,
                          pixels_per_metre: float = None) -> dict[int, float]:
    """
    Compute instantaneous speed for each track visible near frame_num.
    Uses a sliding window of `window` frames for smoothing.
    """
    speeds = {}
    for tid, pts in positions_by_id.items():
        recent = [(f, x, y) for f, x, y in pts
                  if frame_num - window <= f <= frame_num]
        if len(recent) < 2:
            continue
        f0, x0, y0 = recent[0]
        f1, x1, y1 = recent[-1]
        dt = max(1, f1 - f0)
        dist_px = np.hypot(x1 - x0, y1 - y0)
        speed = dist_px / dt * fps
        if pixels_per_metre:
            speed /= pixels_per_metre
        speeds[tid] = round(speed, 1)
    return speeds


# ── Video overlay ─────────────────────────────────────────────────────────────

def produce_speed_video(track_csv: str, video_path: str,
                         output_path: str, pixels_per_metre: float = None):
    """Annotate each bounding box with live speed estimate."""
    # Load track data
    track_data = []
    with open(track_csv) as f:
        for row in csv.DictReader(f):
            track_data.append({k: int(v) for k, v in row.items()})

    # Index by frame and by track_id
    by_frame: dict[int, list] = defaultdict(list)
    by_id:    dict[int, list] = defaultdict(list)
    for r in track_data:
        by_frame[r["frame"]].append(r)
        by_id[r["track_id"]].append((r["frame"], r["cx"], r["cy"]))

    cap = cv2.VideoCapture(video_path)
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Try codecs: mp4v (most compatible) -> XVID -> avc1 (H.264, may fail on Linux)
    writer = None
    for codec_name in ["mp4v", "XVID", "avc1"]:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"✅ Speed video: VideoWriter opened with codec: {codec_name}")
            break
        writer.release()
        writer = None
    if writer is None:
        print("❌ Error: Could not open VideoWriter for speed video")
        return

    unit_label = "m/s" if pixels_per_metre else "px/s"
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        live_speeds = live_speed_for_frame(
            frame_count, by_id, fps,
            window=12, pixels_per_metre=pixels_per_metre
        )

        for r in by_frame.get(frame_count, []):
            tid   = r["track_id"]
            speed = live_speeds.get(tid, 0.0)

            # Box
            cv2.rectangle(frame,
                          (r["x1"], r["y1"]), (r["x2"], r["y2"]),
                          (0, 220, 255), 2)

            # Speed label
            speed_text = f"ID{tid}  {speed:.1f} {unit_label}"
            cv2.putText(frame, speed_text,
                        (r["x1"], r["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 255), 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✅  Speed video → {output_path}")


# ── Stats CSV ─────────────────────────────────────────────────────────────────

def save_speed_stats(stats: dict, out_path: str):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "avg_speed", "max_speed", "total_distance", "unit"])
        for tid, s in sorted(stats.items()):
            w.writerow([tid, s["avg_speed"], s["max_speed"],
                        s["total_distance"], s["unit"]])
    print(f"📊  Speed stats → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    default="../data/output/track_data.csv")
    p.add_argument("--video",  default="../data/input/video.mp4")
    p.add_argument("--output", default="../data/output/speed_output.mp4")
    p.add_argument("--stats",  default="../data/output/speed_stats.csv")
    p.add_argument("--ppm",    type=float, default=None,
                   help="Pixels per metre calibration (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    track_data = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            track_data.append({k: int(v) for k, v in row.items()})

    cap = cv2.VideoCapture(args.video)
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()

    stats = compute_speeds(track_data, fps, pixels_per_metre=args.ppm)
    save_speed_stats(stats, args.stats)

    unit = "m/s" if args.ppm else "px/s"
    print(f"\n📈  Top movers ({unit}):")
    top = sorted(stats.items(), key=lambda x: x[1]["avg_speed"], reverse=True)[:5]
    for tid, s in top:
        print(f"   ID {tid:3d}  avg={s['avg_speed']:6.1f}  max={s['max_speed']:6.1f}  dist={s['total_distance']:7.1f}")

    produce_speed_video(args.csv, args.video, args.output, pixels_per_metre=args.ppm)