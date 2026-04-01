import streamlit as st
import os
import subprocess
import sys
import pandas as pd
import warnings

# Suppress pkg_resources deprecation warning from deep_sort_realtime
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

os.environ["PYTHONIOENCODING"] = "utf-8"

st.set_page_config(page_title="⚽ Tracking Dashboard", layout="wide")
st.title("⚽ Multi-Object Detection & Persistent ID Tracking")

# ─────────────────────────────────────────────
# PATHS  (all absolute, derived from this file)
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
SRC_DIR    = os.path.join(BASE_DIR, "src")
PYTHON     = sys.executable

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────
@st.cache_resource
def verify_dependencies():
    missing = []
    checks = {
        "cv2":                "opencv-python-headless",
        "ultralytics":        "ultralytics",
        "deep_sort_realtime": "deep-sort-realtime",
    }
    for module, pkg in checks.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    return missing

missing_pkgs = verify_dependencies()
if missing_pkgs:
    st.error(
        f"Missing packages: `{', '.join(missing_pkgs)}`\n\n"
        f"Run this in your venv then restart:\n"
        f"```\n{PYTHON} -m pip install {' '.join(missing_pkgs)}\n```"
    )
    st.stop()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def show_video(filename, title):
    path = os.path.join(OUTPUT_DIR, filename)
    st.subheader(title)
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size < 1000:
            st.warning(f"File exists but may be corrupt ({size} bytes): `{filename}`")
        else:
            st.video(path)
    else:
        st.warning(f"Not found: `{filename}`")
        if os.path.exists(OUTPUT_DIR):
            files = os.listdir(OUTPUT_DIR)
            st.info(f"Files currently in output dir: {files}")

def show_image(filename, title):
    path = os.path.join(OUTPUT_DIR, filename)
    st.subheader(title)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Not found: `{filename}`")

def show_csv(filename, title):
    path = os.path.join(OUTPUT_DIR, filename)
    st.subheader(title)
    if os.path.exists(path):
        st.dataframe(pd.read_csv(path), use_container_width=True)
    else:
        st.warning(f"Not found: `{filename}`")

def run_step(label, cmd, cwd):
    st.write(f"▶ Running: {label}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with st.expander(f"Logs — {label}", expanded=(result.returncode != 0)):
        st.text(f"Return code : {result.returncode}")
        st.text(f"Working dir : {cwd}")
        st.text(f"Command     : {' '.join(cmd)}")
        if result.stdout:
            st.text(result.stdout[-4000:])
        if result.stderr:
            st.error(result.stderr[-4000:])
    return result.returncode == 0

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Options")
    conf        = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)
    min_height  = st.number_input("Min bbox height (px)", 50, 300, 120, 10)
    frame_skip  = st.selectbox("Frame skip (1 = every frame)", [1, 2, 3], index=1)
    trail       = st.number_input("Trail length (frames)", 10, 100, 40, 5)
    ppm         = st.number_input("Pixels per metre (0 = px/s)", 0.0, 50.0, 0.0, 0.5)
    run_compare = st.checkbox("Run YOLOv8n vs YOLOv8s comparison", value=True)
    st.divider()
    st.caption("Activate your venv before launching Streamlit.")

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a sports video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    input_path = os.path.join(INPUT_DIR, "video.mp4")

    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Video saved → `{input_path}`")
    except Exception as e:
        st.error(f"Could not save video: {e}")
        st.stop()

    if st.button("Run Full Pipeline"):

        with st.status("Running pipeline...", expanded=True) as status:

            # ── Step 1: Core detection + tracking ──
            step1_ok = run_step(
                "Core detection + tracking (main.py)",
                [
                    PYTHON, "main.py",
                    "--input",      input_path,
                    "--output",     os.path.join(OUTPUT_DIR, "output.mp4"),
                    "--conf",       str(conf),
                    "--min-height", str(int(min_height)),
                    "--frame-skip", str(frame_skip),
                    "--trail",      str(int(trail)),
                    "--no-display",
                ],
                cwd=SRC_DIR,
            )

            if not step1_ok:
                status.update(label="Core pipeline failed — check logs above.", state="error")
                st.stop()

            # ── Step 2: All enhancements ──
            enhance_cmd = [
                PYTHON, "run_all_enhancements.py",
                "--video",   input_path,
                "--out-dir", OUTPUT_DIR,
            ]
            if ppm > 0:
                enhance_cmd += ["--ppm", str(ppm)]
            if run_compare:
                enhance_cmd += ["--compare"]

            step2_ok = run_step(
                "Enhancements (speed, team, bird's-eye, evaluation)",
                enhance_cmd,
                cwd=SRC_DIR,
            )

            if not step2_ok:
                status.update(label="Enhancements had errors — partial results shown below.", state="error")
            else:
                status.update(label="Pipeline complete!", state="complete")

        st.divider()

        # ─────────────────────────────────────────
        # RESULTS
        # ─────────────────────────────────────────

        st.header("Tracking Output")
        show_video("output.mp4", "Annotated Tracking Video")

        st.header("Spatial Analytics")
        col1, col2 = st.columns(2)
        with col1:
            show_image("heatmap.png", "Movement Heatmap")
        with col2:
            show_image("trajectory_map.png", "Trajectory Map")

        st.header("Count & Lifetime Analytics")
        col3, col4 = st.columns(2)
        with col3:
            show_image("count_plot.png", "People Count Over Time")
        with col4:
            show_image("track_lifetime_plot.png", "Track Lifetime")

        st.header("Team Clustering")
        show_video("team_output.mp4", "Team-Coloured Tracking Video")

        st.header("Speed Estimation")
        show_video("speed_output.mp4", "Live Speed Overlay Video")
        show_csv("speed_stats.csv", "Per-Track Speed Stats")

        st.header("Bird's-Eye View")
        show_video("birds_eye.mp4", "Top-Down Projection")

        st.header("Model Comparison")
        show_image("model_comparison.png", "YOLOv8n vs YOLOv8s")

        st.header("Raw Tracking Data")
        show_csv("track_data.csv",      "Full Track Data")
        show_csv("count_over_time.csv", "Person Count Per Frame")