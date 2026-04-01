"""
birds_eye.py
------------
Projects tracked player positions onto a top-down (bird's-eye) court view
using a homography computed from visible court lines.

Two modes:
  A) AUTO  – tries to detect the court boundary automatically via colour
             thresholding of the court floor.
  B) MANUAL – you supply 4 source points (corners visible in the camera
              view) and 4 destination points (on the flat court diagram).

The output is a side-by-side video:
  LEFT  = original annotated frame
  RIGHT = bird's-eye court diagram with coloured dots per player

Usage:
    # Manual mode (recommended for broadcast footage)
    python birds_eye.py \
        --csv  ../data/output/track_data.csv \
        --video ../data/input/video.mp4 \
        --output ../data/output/birds_eye.mp4 \
        --src-pts "50,420,1230,420,1230,50,50,50" \
        --dst-pts "0,400,400,400,400,0,0,0"

    # Auto mode (works if court colour is distinctive)
    python birds_eye.py --auto
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

# ── Standard basketball half-court diagram dimensions (pixels) ───────────────
COURT_W, COURT_H = 470, 500   # width × height of the flat court diagram


def build_court_diagram() -> np.ndarray:
    """Draw a simple basketball half-court diagram."""
    court = np.full((COURT_H, COURT_W, 3), 30, dtype=np.uint8)  # dark bg
    lc    = (60, 160, 60)    # line colour (green)
    lw    = 2

    # Court boundary
    cv2.rectangle(court, (10, 10), (COURT_W - 10, COURT_H - 10), lc, lw)

    # Centre circle area (simplified)
    cv2.ellipse(court, (COURT_W // 2, COURT_H - 10), (60, 60),
                0, 180, 360, lc, lw)

    # Key / paint area
    key_x1 = COURT_W // 2 - 60
    key_x2 = COURT_W // 2 + 60
    cv2.rectangle(court, (key_x1, COURT_H - 10), (key_x2, COURT_H - 180), lc, lw)

    # Free-throw circle
    cv2.circle(court, (COURT_W // 2, COURT_H - 180), 60, lc, lw)

    # Basket
    cv2.circle(court, (COURT_W // 2, COURT_H - 30), 10, (0, 180, 255), 2)

    # Three-point arc (approximate)
    cv2.ellipse(court, (COURT_W // 2, COURT_H - 30), (160, 140),
                0, 180, 360, lc, lw)

    return court


def parse_pts(s: str) -> np.ndarray:
    """Parse comma-separated x,y,x,y,... into (N,2) float32 array."""
    vals = list(map(float, s.split(",")))
    return np.array(vals, dtype=np.float32).reshape(-1, 2)


def auto_detect_court_pts(frame: np.ndarray):
    """
    Very rough court detection by finding the largest light-coloured
    quadrilateral (wood floor colour range in HSV).
    Returns 4 source points or None.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Wood floor: low saturation, medium-high value
    mask = cv2.inRange(hsv, (0, 0, 120), (30, 60, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    peri    = cv2.arcLength(biggest, True)
    approx  = cv2.approxPolyDP(biggest, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        # Sort: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff    = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    return None


def produce_birds_eye_video(
    track_csv:  str,
    video_path: str,
    output_path: str,
    src_pts:    np.ndarray | None = None,
    auto_mode:  bool = False,
):
    """
    Produce a side-by-side (original | bird's-eye) video.

    Args:
        track_csv:   Path to track_data.csv from main.py.
        video_path:  Original input video.
        output_path: Where to save the output.
        src_pts:     4 points in the camera frame (court corners).
                     If None and not auto_mode, a default estimate is used.
        auto_mode:   Attempt automatic court detection from first frame.
    """
    # Load track data
    track_data = []
    with open(track_csv) as f:
        for row in csv.DictReader(f):
            track_data.append({k: int(v) for k, v in row.items()})

    by_frame: dict[int, list] = defaultdict(list)
    for r in track_data:
        by_frame[r["frame"]].append(r)

    cap = cv2.VideoCapture(video_path)
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Compute homography ────────────────────────────────────────────────
    dst_pts = np.array([
        [10,          10         ],
        [COURT_W - 10, 10        ],
        [COURT_W - 10, COURT_H-10],
        [10,          COURT_H - 10],
    ], dtype=np.float32)

    if auto_mode or src_pts is None:
        ret, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if auto_mode and ret:
            detected = auto_detect_court_pts(first_frame)
            src_pts  = detected

        if src_pts is None:
            # Fallback: use full frame corners (rough but always works)
            src_pts = np.array([
                [fw * 0.05, fh * 0.15],
                [fw * 0.95, fh * 0.15],
                [fw * 0.95, fh * 0.90],
                [fw * 0.05, fh * 0.90],
            ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)

    # ── Output video (side-by-side) ───────────────────────────────────────
    out_w  = fw + COURT_W
    out_h  = max(fh, COURT_H)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Try avc1 first, fall back to mp4v, then XVID
    writer = None
    for codec_name in ["avc1", "mp4v", "XVID"]:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if writer.isOpened():
            print(f"✅ Bird's-eye video: VideoWriter opened with codec: {codec_name}")
            break
        writer.release()
        writer = None
    if writer is None:
        print("❌ Error: Could not open VideoWriter for bird's-eye video")
        return

    court_base = build_court_diagram()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        court = court_base.copy()

        for r in by_frame.get(frame_count, []):
            tid = r["track_id"]
            # Foot point in camera space
            fx, fy = (r["x1"] + r["x2"]) // 2, r["y2"]

            # Project to court space
            pt  = np.array([[[float(fx), float(fy)]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pt, H)
            px, py = int(dst[0][0][0]), int(dst[0][0][1])

            # Pick colour based on hue (simple hash)
            np.random.seed(tid * 137)
            color = tuple(int(c) for c in np.random.randint(100, 255, 3))

            if 0 <= px < COURT_W and 0 <= py < COURT_H:
                cv2.circle(court, (px, py), 8, color, -1)
                cv2.putText(court, str(tid), (px + 9, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Also draw on original frame
            cv2.rectangle(frame, (r["x1"], r["y1"]), (r["x2"], r["y2"]),
                          color, 2)

        # Side-by-side composite
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:fh,   :fw]       = frame
        canvas[:COURT_H, fw:]    = court

        # Label panels
        cv2.putText(canvas, "Camera View", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(canvas, "Bird's-Eye View", (fw + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        writer.write(canvas)

    cap.release()
    writer.release()
    print(f"✅  Bird's-eye video → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      default="../data/output/track_data.csv")
    p.add_argument("--video",    default="../data/input/video.mp4")
    p.add_argument("--output",   default="../data/output/birds_eye.mp4")
    p.add_argument("--src-pts",  default=None,
                   help="8 comma-separated values: x1,y1,x2,y2,x3,y3,x4,y4")
    p.add_argument("--auto",     action="store_true",
                   help="Auto-detect court boundary")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    src_pts = parse_pts(args.src_pts) if args.src_pts else None
    produce_birds_eye_video(
        args.csv, args.video, args.output,
        src_pts=src_pts, auto_mode=args.auto
    )