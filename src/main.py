"""
main.py
-------
Entry point for the Multi-Object Detection and Tracking pipeline.

Usage:
    python main.py
    python main.py --input ../data/input/video.mp4 --output ../data/output/output.mp4
    python main.py --no-display --frame-skip 2

Outputs:
    - Annotated output video
    - heatmap.png
    - trajectory_map.png
    - count_over_time.csv
    - track_data.csv  (per-frame positions for all enhancements)
"""
import argparse
import csv
import os

import cv2
import numpy as np

from detector import Detector
from tracker import Tracker
from visualize import HeatmapAccumulator, draw_tracks, reset_history, get_color

DEFAULT_INPUT = "data/input/video.mp4"
DEFAULT_OUTPUT = "data/output/output.mp4"


def parse_args():
    p = argparse.ArgumentParser(description="Multi-Object Tracking Pipeline")
    p.add_argument("--input",      default=DEFAULT_INPUT)
    p.add_argument("--output",     default=DEFAULT_OUTPUT)
    p.add_argument("--model",      default="yolov8n.pt")
    p.add_argument("--conf",       type=float, default=0.5)
    p.add_argument("--min-height", type=int,   default=120)
    p.add_argument("--frame-skip", type=int,   default=2)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--trail",      type=int,   default=40)
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {args.input}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹  {frame_w}×{frame_h} @ {fps} fps  |  ~{total} frames")

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    # Try avc1 (H.264) first, fall back to mp4v, then XVID
    safe_fps = max(10, fps)
    writer = None
    for codec_name in ["avc1", "mp4v", "XVID"]:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(args.output, fourcc, safe_fps, (frame_w, frame_h))
        if writer.isOpened():
            print(f"✅ VideoWriter opened with codec: {codec_name}")
            break
        writer.release()
        writer = None

    if writer is None:
        print("❌ Error: Could not open VideoWriter with any codec")
        return
    detector  = Detector(args.model, conf_threshold=args.conf, min_height=args.min_height)
    tracker   = Tracker()
    heatmap   = HeatmapAccumulator(frame_h, frame_w)
    reset_history()

    # --- data collectors for enhancements ---
    count_log   = []                          # (frame, count)
    track_log   = []                          # (frame, id, cx, cy, x1,y1,x2,y2)
    last_frame  = None
    frame_count = 0

    print("🚀  Processing …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % args.frame_skip != 0:
            continue

        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame)

        # collect positions
        for x1, y1, x2, y2, tid in tracks:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_log.append((frame_count, tid, cx, cy, x1, y1, x2, y2))

        count_log.append((frame_count, len(tracks)))
        heatmap.update(tracks)

        frame = draw_tracks(frame, tracks, trail_length=args.trail)
        writer.write(frame)
        last_frame = frame.copy()

        if not args.no_display:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if frame_count % 100 == 0:
            print(f"   Frame {frame_count}/{total} …")

    cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Save heatmap ──────────────────────────────────────────────────────
    if last_frame is not None:
        heat_img = heatmap.render(background=last_frame, alpha=0.6)
        cv2.imwrite(os.path.join(out_dir, "heatmap.png"), heat_img)
        print(f"🌡️   Heatmap          → {out_dir}/heatmap.png")

    # ── Save trajectory map ───────────────────────────────────────────────
    if last_frame is not None and track_log:
        traj_img = draw_trajectory_map(track_log, frame_h, frame_w, last_frame)
        cv2.imwrite(os.path.join(out_dir, "trajectory_map.png"), traj_img)
        print(f"🗺️   Trajectory map   → {out_dir}/trajectory_map.png")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    csv_count = os.path.join(out_dir, "count_over_time.csv")
    with open(csv_count, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "person_count"])
        w.writerows(count_log)

    csv_tracks = os.path.join(out_dir, "track_data.csv")
    with open(csv_tracks, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "cx", "cy", "x1", "y1", "x2", "y2"])
        w.writerows(track_log)

    print(f"📊  Count CSV        → {csv_count}")
    print(f"📋  Track data CSV   → {csv_tracks}")
    print("✅  Done!")
    print(f"📁  Output video     → {args.output}")


def draw_trajectory_map(track_log, frame_h, frame_w, background=None):
    """Draw all trajectories on a single image (one colour per ID)."""
    from collections import defaultdict
    canvas = background.copy() if background is not None else np.zeros((frame_h, frame_w, 3), np.uint8)
    # darken background so trails stand out
    canvas = (canvas * 0.35).astype(np.uint8)

    positions = defaultdict(list)
    for frame, tid, cx, cy, *_ in track_log:
        positions[tid].append((cx, cy))

    for tid, pts in positions.items():
        color = get_color(tid)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], color, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 5, color, -1)
            cv2.putText(canvas, f"ID{tid}", (pts[-1][0]+6, pts[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    main()