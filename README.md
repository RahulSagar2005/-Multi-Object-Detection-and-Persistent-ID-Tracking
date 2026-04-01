# Multi-Object Detection & Persistent ID Tracking

> **Assignment:** Multi-Object Detection and Persistent ID Tracking in Public Sports/Event Footage  
> **Tech stack:** YOLOv8n · DeepSORT · OpenCV · Python 3.10+

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Installation](#3-installation)
4. [How to Run — Core Pipeline](#4-how-to-run--core-pipeline)
5. [How to Run — All Optional Enhancements](#5-how-to-run--all-optional-enhancements)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [Module Reference](#7-module-reference)
8. [Optional Enhancements](#8-optional-enhancements)
9. [Model & Tracker Choices](#9-model--tracker-choices)
10. [Assumptions](#10-assumptions)
11. [Limitations](#11-limitations)
12. [All Output Files](#12-all-output-files)
13. [Video Source](#13-video-source)

---

## 1. Project Overview

End-to-end computer vision pipeline that:
- **Detects** every person in a sports video using YOLOv8
- **Tracks** each person with a unique persistent ID using DeepSORT
- **Visualises** results with bounding boxes, ID labels, and motion trails
- **Exports** a rich set of analytics: heatmap, trajectory map, speed stats, team clustering, bird's-eye view, and evaluation metrics

---

## 2. Folder Structure

```
project/
├── data/
│   ├── input/
│   │   └── video.mp4                  ← place your input video here
│   └── output/
│       ├── output.mp4                 ← annotated tracking video
│       ├── heatmap.png                ← movement density heatmap
│       ├── trajectory_map.png         ← all trajectories overlaid
│       ├── count_over_time.csv        ← person count per frame
│       ├── track_data.csv             ← full per-frame bounding boxes + IDs
│       ├── speed_stats.csv            ← per-track speed statistics
│       ├── speed_output.mp4           ← live speed overlay video
│       ├── team_output.mp4            ← team-coloured tracking video
│       ├── birds_eye.mp4              ← side-by-side bird's-eye video
│       ├── count_plot.png             ← person count chart
│       ├── track_lifetime_plot.png    ← track duration charts
│       ├── model_comparison.png       ← YOLOv8n vs YOLOv8s chart
│       └── team_split.png             ← team distribution bar chart
├── src/
│   ├── main.py                        ← entry point (core pipeline)
│   ├── detector.py                    ← YOLOv8 wrapper
│   ├── tracker.py                     ← DeepSORT wrapper
│   ├── visualize.py                   ← drawing + heatmap
│   ├── team_clustering.py             ← jersey-colour KMeans clustering
│   ├── speed_estimation.py            ← per-track speed analytics
│   ├── birds_eye.py                   ← homography top-down projection
│   ├── evaluation.py                  ← metrics + model comparison
│   ├── run_all_enhancements.py        ← one-command enhancement runner
│   └── yolov8n.pt                     ← model weights (auto-downloaded)
├── notebook.ipynb                     ← full interactive demo
├── requirements.txt
├── README.md
└── TECHNICAL_REPORT.md
```

---

## 3. Installation

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

> **YOLOv8 weights** (`yolov8n.pt`) are auto-downloaded on first run if not present.  
> Place the provided `yolov8n.pt` inside `src/` to skip the download.

---

## 4. How to Run — Core Pipeline

### Basic
```bash
cd src
python main.py
```

### With options
```bash
python main.py \
  --input        ../data/input/video.mp4 \
  --output       ../data/output/output.mp4 \
  --model        yolov8n.pt \
  --conf         0.5 \
  --min-height   120 \
  --frame-skip   2 \
  --trail        40 \
  --no-display
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `../data/input/video.mp4` | Input video |
| `--output` | `../data/output/output.mp4` | Output annotated video |
| `--model` | `yolov8n.pt` | YOLOv8 weights |
| `--conf` | `0.5` | Detection confidence threshold |
| `--min-height` | `120` | Min bbox height px (removes audience) |
| `--frame-skip` | `2` | Process every N-th frame |
| `--trail` | `40` | Trajectory trail length |
| `--no-display` | off | Headless / server mode |

**Outputs produced by `main.py`:**
- `output.mp4` — annotated video with boxes, IDs, trails, count
- `heatmap.png` — foot-position density map
- `trajectory_map.png` — all trajectories on darkened background
- `count_over_time.csv` — person count per frame
- `track_data.csv` — full tracking data (used by all enhancements)

---

## 5. How to Run — All Optional Enhancements

### One command (recommended)
```bash
cd src
python run_all_enhancements.py
```

### With calibration + model comparison
```bash
python run_all_enhancements.py --ppm 8.5 --compare
```

### Skip individual steps
```bash
python run_all_enhancements.py --skip-birds-eye --skip-team
```

### Run each enhancement individually
```bash
# Speed estimation
python speed_estimation.py --ppm 8.5

# Team clustering video
python team_clustering.py

# Bird's-eye projection
python birds_eye.py --auto

# Evaluation metrics + plots
python evaluation.py

# Evaluation + model comparison
python evaluation.py --compare --video ../data/input/video.mp4
```

### Jupyter notebook
```bash
cd ..          # project root
jupyter notebook notebook.ipynb
```
Run all cells in order — each section produces inline charts.

---

## 6. Pipeline Architecture

```
Input Video
     │
     ▼  (every frame_skip-th frame)
┌─────────────────────────────────────────────────────┐
│  Detector  (YOLOv8n)                                │
│  • Inference @ conf ≥ 0.5                           │
│  • Class 0 (person) only                            │
│  • Height filter: < 120 px  → discard               │
│  • Aspect ratio: h < 0.5×w  → discard               │
└──────────────────────┬──────────────────────────────┘
                       │ [(bbox, conf, "person"), …]
                       ▼
┌─────────────────────────────────────────────────────┐
│  Tracker  (DeepSORT)                                │
│  • Kalman Filter  → motion prediction               │
│  • CNN Re-ID      → appearance matching             │
│  • Hungarian Alg  → optimal assignment              │
│  • n_init=5       → confirm after 5 hits            │
│  • max_age=50     → survive 50 frames lost          │
└──────────────────────┬──────────────────────────────┘
                       │ [(x1,y1,x2,y2, track_id), …]
                       ▼
┌─────────────────────────────────────────────────────┐
│  Visualiser  (visualize.py)                         │
│  • Coloured bounding boxes (unique per ID)          │
│  • ID label with filled background                  │
│  • Fading trajectory trail (last 40 points)         │
│  • People count overlay                             │
│  • HeatmapAccumulator (foot positions)              │
└──────────────────────┬──────────────────────────────┘
                       │
     ┌─────────────────┼──────────────────┐
     ▼                 ▼                  ▼
output.mp4       heatmap.png        track_data.csv
                 trajectory_map.png count_over_time.csv
                       │
                       ▼  (run_all_enhancements.py)
     ┌─────────────────┬──────────────────┬──────────────┐
     ▼                 ▼                  ▼              ▼
speed_output.mp4  team_output.mp4  birds_eye.mp4  count_plot.png
speed_stats.csv   team_split.png   (side-by-side) model_comparison.png
```

---

## 7. Module Reference

### `detector.py`
YOLOv8 wrapper. Filters by class (person only), confidence, minimum height, and aspect ratio.

### `tracker.py`
DeepSORT wrapper. Configured with `max_age=50`, `n_init=5`, `max_cosine_distance=0.2`.

### `visualize.py`
- `draw_tracks()` — draws boxes, ID labels, fading trajectory trails, person count
- `HeatmapAccumulator` — accumulates foot positions; `render()` produces a JET colourmap overlay
- `reset_history()` — clears state between videos

### `team_clustering.py`
- Extracts HSV hue histograms from upper-body crops (jersey region)
- KMeans(k=2) clusters players into Team A / Team B
- `TeamClusterer.update()` — feeds one frame/crop
- `TeamClusterer.assign_teams()` — triggers clustering (call every ~60 frames)
- `demo_from_csv()` — produces a full team-annotated video from `track_data.csv`

### `speed_estimation.py`
- `compute_speeds()` — sliding-window pixel displacement → px/s or m/s
- `produce_speed_video()` — overlays live speed label on each bbox
- `save_speed_stats()` — writes per-track CSV summary
- `--ppm` flag: pixels per metre calibration (measure a known distance in the frame)

### `birds_eye.py`
- Computes a homography from camera-frame court corners → flat court diagram
- `--auto` flag: attempts colour-threshold court detection on first frame
- `--src-pts`: manual 4-point specification (most accurate for broadcast footage)
- Output: side-by-side video (original | top-down court with coloured dots)

### `evaluation.py`
- **Tracklet metrics** (no ground truth needed):
  - Unique IDs assigned
  - Average / min / max track lifetime
  - Average detections per frame
  - Fragmentation ratio (proxy for ID switch rate)
- **Plots**: track lifetime histogram, count-over-time chart
- **Model comparison**: YOLOv8n vs YOLOv8s on N sample frames
  - Inference time (ms/frame)
  - Detection count (persons @ conf≥0.5)
  - Confidence distribution

### `run_all_enhancements.py`
Single entry point that runs steps 1-4 above in sequence with error handling and prerequisite checks.

---

## 8. Optional Enhancements

| Enhancement | Status | File | Output |
|---|---|---|---|
| Trajectory visualisation | ✅ | `main.py` + `visualize.py` | `trajectory_map.png` + trails on video |
| Movement heatmaps | ✅ | `visualize.py` | `heatmap.png` |
| Bird's-eye projection | ✅ | `birds_eye.py` | `birds_eye.mp4` |
| Object count over time | ✅ | `main.py` + `evaluation.py` | `count_plot.png` + CSV |
| Team / role clustering | ✅ | `team_clustering.py` | `team_output.mp4` + `team_split.png` |
| Speed estimation | ✅ | `speed_estimation.py` | `speed_output.mp4` + `speed_stats.csv` |
| Simple evaluation metrics | ✅ | `evaluation.py` | `track_lifetime_plot.png` + printed stats |
| Model comparison | ✅ | `evaluation.py` | `model_comparison.png` |
| Notebook / demo script | ✅ | `notebook.ipynb` | Inline charts for all of the above |

---

## 9. Model & Tracker Choices

### Detector — YOLOv8n
- Single-stage detector; processes entire image in one forward pass
- Pre-trained on COCO (80 classes); class 0 = person
- Nano variant chosen for CPU speed; swap `--model yolov8s.pt` for higher accuracy
- Two post-detection filters prevent audience/noise detections

### Tracker — DeepSORT
| Component | Role |
|---|---|
| Kalman Filter | Predicts next bbox location using constant-velocity model |
| CNN Re-ID (128-d) | Distinguishes similar-looking players via appearance embeddings |
| Hungarian Algorithm | Optimal O(n³) detection-to-track assignment |

**Parameter rationale:**
- `max_age=50` — survives ~1.5 s occlusion at 30 fps (typical sports play)
- `n_init=5` — prevents noise/shadow detections becoming confirmed tracks
- `max_cosine_distance=0.2` — strict appearance matching; reduces same-team ID confusion

---

## 10. Assumptions

1. Input is a standard MP4 / H.264 file readable by OpenCV.
2. Subjects of interest are people (COCO class 0). Change `allowed_classes` in `detector.py` for other objects.
3. Bounding boxes shorter than 120 px are audience/background — too small to track reliably.
4. `frame_skip=2` provides good speed/accuracy balance. Use `1` for maximum accuracy.
5. No GPU assumed — runs on CPU. Add `device='cuda'` to `YOLO()` call for GPU.
6. Team clustering assumes 2 teams with visually distinct jersey colours.
7. Speed in `px/s` unless `--ppm` calibration is provided.

---

## 11. Limitations

| Limitation | Impact |
|---|---|
| Severe occlusion (> 50 frames) | Track lost; re-assigned new ID when reappearing |
| Identical jerseys within a team | Appearance Re-ID may briefly swap IDs |
| Rapid camera panning/zoom | Kalman prediction degrades without camera-motion compensation |
| Very crowded scenes (> 10 overlap) | Merged bboxes cause temporary ID confusion |
| Bird's-eye auto-mode | Court detection by colour may fail on unusual court surfaces |
| Team clustering | Requires enough frames to build feature gallery; referees may cluster as a third group |

---

## 12. All Output Files

| File | Description |
|---|---|
| `output.mp4` | Annotated tracking video (boxes + IDs + trails) |
| `heatmap.png` | Movement density heatmap (JET colourmap) |
| `trajectory_map.png` | All player trajectories on darkened background |
| `count_over_time.csv` | Frame-by-frame person count |
| `track_data.csv` | Full tracking data: frame, id, cx, cy, x1, y1, x2, y2 |
| `speed_output.mp4` | Live speed label overlaid on each tracked bbox |
| `speed_stats.csv` | Per-track: avg speed, max speed, total distance |
| `team_output.mp4` | Team A / Team B colour-coded tracking video |
| `birds_eye.mp4` | Side-by-side: camera view + top-down court projection |
| `count_plot.png` | Person count over time with rolling average |
| `track_lifetime_plot.png` | Track duration histogram + rank chart |
| `model_comparison.png` | YOLOv8n vs YOLOv8s: speed, count, confidence |
| `team_split.png` | Bar chart of players per team |

---

## 13. Video Source

> **Public video URL:** *(add your chosen video link here)*

Download with yt-dlp:
```bash
pip install yt-dlp
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" <URL> -o data/input/video.mp4
```

---

## Dependencies

```
ultralytics>=8.0.0        # YOLOv8
deep-sort-realtime>=1.3.2 # DeepSORT
opencv-python>=4.8.0      # Video I/O
numpy>=1.24.0
matplotlib>=3.7.0         # Plots
scikit-learn>=1.3.0       # KMeans team clustering
scipy>=1.11.0
pandas>=2.0.0
seaborn>=0.12.0
jupyter>=1.0.0
```
