"""
run_all_enhancements.py
------------------------
Runs ALL optional enhancements in sequence after main.py has produced:
  - data/output/track_data.csv
  - data/output/count_over_time.csv
  - data/input/video.mp4

Usage (from repo root OR from src/):
    python src/run_all_enhancements.py
    python run_all_enhancements.py --ppm 8.5 --compare
    python run_all_enhancements.py --skip-birds-eye --skip-team

All paths are resolved relative to THIS FILE so the script works
regardless of the current working directory (local or Streamlit Cloud).
"""

import argparse
import csv
import os
import sys

import cv2

# ─────────────────────────────────────────────────────────────────────────────
# ABSOLUTE PATHS  (derived from this file's location, never from cwd)
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))   # …/project/src
_ROOT = os.path.dirname(_HERE)                        # …/project

_DEFAULT_TRACK_CSV = os.path.join(_ROOT, "data", "output", "track_data.csv")
_DEFAULT_COUNT_CSV = os.path.join(_ROOT, "data", "output", "count_over_time.csv")
_DEFAULT_VIDEO     = os.path.join(_ROOT, "data", "input",  "video.mp4")
_DEFAULT_OUT_DIR   = os.path.join(_ROOT, "data", "output")


# ── Prerequisites ─────────────────────────────────────────────────────────────
def check_prerequisites(track_csv, count_csv, video):
    missing = []
    for path in [track_csv, count_csv, video]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("❌  Missing files (run main.py first):")
        for p in missing:
            print(f"     {p}")
        sys.exit(1)
    print("✅  Prerequisites found\n")


def _load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: int(v) for k, v in row.items()})
    return rows


def _get_fps(video):
    cap = cv2.VideoCapture(video)
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    return fps


# ── 1. Evaluation Metrics + Count Plot ───────────────────────────────────────
def run_evaluation(track_csv, count_csv, out_dir, compare=False, video=None):
    print("━" * 55)
    print("  1/4  EVALUATION METRICS + PLOTS")
    print("━" * 55)
    from evaluation import (
        load_track_data, compute_tracklet_metrics,
        print_metrics, plot_lifetime_histogram,
        plot_count_over_time, compare_models,
    )

    data    = load_track_data(track_csv)
    metrics = compute_tracklet_metrics(data)
    print_metrics(metrics)

    plot_lifetime_histogram(
        metrics["track_lifetimes"],
        os.path.join(out_dir, "track_lifetime_plot.png"),
    )
    plot_count_over_time(
        count_csv,
        os.path.join(out_dir, "count_plot.png"),
    )

    if compare and video:
        print("\n🔬  Running model comparison (YOLOv8n vs YOLOv8s) …")
        compare_models(
            video, n_frames=80,
            out_path=os.path.join(out_dir, "model_comparison.png"),
        )


# ── 2. Speed Estimation ───────────────────────────────────────────────────────
def run_speed(track_csv, out_dir, ppm=None, video=None):
    print("\n" + "━" * 55)
    print("  2/4  SPEED ESTIMATION")
    print("━" * 55)
    from speed_estimation import compute_speeds, save_speed_stats, produce_speed_video

    track_data = _load_csv(track_csv)
    fps        = _get_fps(video)

    stats = compute_speeds(track_data, fps, pixels_per_metre=ppm)
    save_speed_stats(stats, os.path.join(out_dir, "speed_stats.csv"))

    unit = "m/s" if ppm else "px/s"
    print(f"\n📈  Top movers ({unit}):")
    top = sorted(stats.items(), key=lambda x: x[1]["avg_speed"], reverse=True)[:5]
    for tid, s in top:
        print(f"   ID {tid:3d}  avg={s['avg_speed']:6.1f}  "
              f"max={s['max_speed']:6.1f}  dist={s['total_distance']:7.1f}")

    produce_speed_video(
        track_csv, video,
        os.path.join(out_dir, "speed_output.mp4"),
        pixels_per_metre=ppm,
    )


# ── 3. Team Clustering ────────────────────────────────────────────────────────
def run_team_clustering(track_csv, out_dir, video=None):
    print("\n" + "━" * 55)
    print("  3/4  TEAM CLUSTERING")
    print("━" * 55)
    from team_clustering import demo_from_csv
    demo_from_csv(
        track_csv, video,
        os.path.join(out_dir, "team_output.mp4"),
    )


# ── 4. Bird's-Eye Projection ──────────────────────────────────────────────────
def run_birds_eye(track_csv, out_dir, video=None):
    print("\n" + "━" * 55)
    print("  4/4  BIRD'S-EYE PROJECTION")
    print("━" * 55)
    from birds_eye import produce_birds_eye_video
    produce_birds_eye_video(
        track_csv, video,
        os.path.join(out_dir, "birds_eye.mp4"),
        src_pts=None,
        auto_mode=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run all optional enhancements")
    p.add_argument("--video",          default=_DEFAULT_VIDEO,
                   help="Absolute path to input video")
    p.add_argument("--out-dir",        default=_DEFAULT_OUT_DIR,
                   help="Absolute path to output directory")
    p.add_argument("--track-csv",      default=_DEFAULT_TRACK_CSV,
                   help="Absolute path to track_data.csv")
    p.add_argument("--count-csv",      default=_DEFAULT_COUNT_CSV,
                   help="Absolute path to count_over_time.csv")
    p.add_argument("--ppm",            type=float, default=None,
                   help="Pixels per metre for speed (e.g. 8.5)")
    p.add_argument("--compare",        action="store_true",
                   help="Run YOLOv8n vs YOLOv8s model comparison")
    p.add_argument("--skip-speed",     action="store_true")
    p.add_argument("--skip-team",      action="store_true")
    p.add_argument("--skip-birds-eye", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve all paths (args override defaults, both are absolute)
    track_csv = args.track_csv
    count_csv = args.count_csv
    video     = args.video
    out_dir   = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    check_prerequisites(track_csv, count_csv, video)

    run_evaluation(track_csv, count_csv, out_dir, compare=args.compare, video=video)

    if not args.skip_speed:
        run_speed(track_csv, out_dir, ppm=args.ppm, video=video)

    if not args.skip_team:
        run_team_clustering(track_csv, out_dir, video=video)

    if not args.skip_birds_eye:
        run_birds_eye(track_csv, out_dir, video=video)

    print("\n" + "=" * 55)
    print("  ✅  ALL ENHANCEMENTS COMPLETE")
    print(f"  📁  Outputs in: {os.path.abspath(out_dir)}")
    print("=" * 55)
    print()
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            icon = "🎬" if f.endswith(".mp4") else "🖼️ " if f.endswith(".png") else "📊"
            print(f"  {icon}  {f:<42} {size/1024:>7.0f} KB")