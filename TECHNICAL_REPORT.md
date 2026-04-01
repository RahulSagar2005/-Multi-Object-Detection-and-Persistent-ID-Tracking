# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Sports Footage

---

## 1. Introduction

This report describes the design, implementation, and evaluation of a computer vision pipeline for detecting and persistently tracking all players in a sports video. The pipeline is built around YOLOv8 for detection and DeepSORT for tracking, extended with a full suite of optional analytics: team clustering, speed estimation, bird's-eye projection, evaluation metrics, and a model comparison study.

---

## 2. Model / Detector Used

**YOLOv8n (Ultralytics)**

YOLO (You Only Look Once) is a single-stage detector that processes the entire image in one forward pass, unlike two-stage methods such as Faster R-CNN that first propose regions then classify them. This makes it significantly faster and suitable for real-time processing. The `n` (nano) variant was selected as the baseline because it runs acceptably on CPU while still achieving strong recall on close-to-camera subjects.

The model is pre-trained on COCO (80 classes). Only class 0 (person) is retained.

**Post-detection filters applied:**

| Filter | Value | Reason |
|--------|-------|--------|
| Confidence threshold | ≥ 0.5 | Removes low-quality detections |
| Minimum bbox height | ≥ 120 px | Removes far-away audience members |
| Aspect ratio | height ≥ 0.5 × width | Removes horizontal false positives |

These filters reduce tracker noise significantly in stadium/court footage where hundreds of audience members are visible in the background.

---

## 3. Tracking Algorithm Used

**DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)**

DeepSORT extends the original SORT tracker by adding a learned appearance descriptor to the assignment step. It combines three components:

**Kalman Filter** predicts each track's next bounding-box location using a constant-velocity motion model in (cx, cy, aspect_ratio, height) space. This allows the tracker to survive frames where a detection is missed (e.g. brief occlusion), because the predicted location is used as a proxy for the real one.

**CNN Re-ID Embeddings** extract a 128-dimensional feature vector from each detection crop. These appearance features are stored in a gallery per track. When assigning detections to tracks, the cosine distance between the new detection's features and the track's gallery is used alongside the Kalman-predicted IoU distance. This is the key difference from plain SORT — it allows the tracker to recover the correct identity after longer occlusions where the Kalman prediction has drifted.

**Hungarian Algorithm** solves the linear assignment problem optimally in O(n³) time, matching the set of detections to the set of active tracks at each frame.

**Parameters chosen:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `max_age` | 50 | Keeps a track alive for 50 frames (~1.5 s at 30 fps) without a match. Sports play regularly involves 1–2 second occlusions. |
| `n_init` | 5 | A track must be confirmed by 5 consecutive detections. Prevents shadows, partial limbs, and crowd noise from creating ghost tracks. |
| `max_cosine_distance` | 0.2 | Strict appearance matching. Reduces the probability of the tracker assigning one team's player to another when they cross paths. |

---

## 4. Why This Combination Was Selected

**Speed:** YOLOv8n achieves ~30–60 fps on modern CPU hardware, making it viable for processing full match footage without a GPU. For even higher accuracy the same pipeline accepts `yolov8s.pt` or `yolov8m.pt` with a single CLI flag change.

**Re-ID capability:** In sports footage, players on the same team often have identical jersey colours and similar builds. A pure motion tracker (SORT, centroid) will swap IDs whenever two such players cross paths. DeepSORT's appearance gallery significantly reduces this by checking whether the candidate detection looks like the expected player, not just whether it appears in the right place.

**Modular design:** Both libraries expose clean Python APIs. The detector and tracker are each wrapped in a single class, making it trivial to swap in alternative detectors (e.g. RT-DETR) or trackers (ByteTrack, StrongSORT) without changing downstream code.

**Community validation:** This combination has been benchmarked extensively on MOTChallenge datasets and sports-specific datasets, providing a solid performance baseline against which improvements can be measured.

---

## 5. How ID Consistency Is Maintained

ID consistency relies on a multi-layered strategy:

**Layer 1 — Motion prediction.** The Kalman Filter maintains a state estimate for each active track. When a detection is missed for one or two frames, the filter extrapolates the track's position. On the next frame, a detection appearing near the predicted location is assigned to the same track, preserving the ID through brief occlusions without requiring any appearance match.

**Layer 2 — Appearance re-identification.** Each track stores a rolling gallery of the last *N* Re-ID feature vectors. When two tracks are spatially close — common when players cross paths — the cosine similarity of their feature galleries is used to break the tie. The detection is assigned to the track whose appearance it most resembles, not simply the spatially closest one.

**Layer 3 — Confirmation buffer.** New tracks start in a "tentative" state for the first `n_init=5` frames. They are not displayed until confirmed. This prevents transient noise (a partial arm, a shadow, a scoreboard element) from spawning a visible ghost track that could collide with a legitimate track's ID space.

**Layer 4 — Generous `max_age`.** Confirmed tracks are not immediately deleted when a detection is lost. They survive for 50 frames in a "lost" state. If the player reappears before the track expires, the same ID is restored, even after a prolonged occlusion.

**Layer 5 — Deterministic colour assignment.** `get_color()` seeds Python's RNG with `track_id × 137`. The same ID always renders the same colour, making any ID switch immediately visible as a sudden colour change in the output video — useful for qualitative evaluation.

---

## 6. Optional Enhancements Implemented

### 6.1 Trajectory Visualisation
Every confirmed track's centre-point history is stored in a deque. In the live video, a fading polyline is drawn from the oldest to the newest point (opacity proportional to recency). After processing, a full trajectory map is saved by darkening the last frame and overlaying all trajectories in their respective track colours.

### 6.2 Movement Heatmap
`HeatmapAccumulator` records the foot position (bottom-centre of each bbox) across all frames by drawing a filled circle of radius 15 px on a float32 accumulation canvas. After processing, the canvas is normalised, colourised with OpenCV's JET colourmap, and blended with the last video frame at α=0.6. The result shows which areas of the court saw the most foot traffic.

### 6.3 Bird's-Eye / Top-Down Projection
A homography matrix H is computed from 4 court-corner correspondences between the camera view and a flat court diagram. At each frame, each player's foot position is projected using `cv2.perspectiveTransform(foot_pt, H)` and plotted as a coloured dot on the diagram. The output is a side-by-side video showing the original annotated frame on the left and the tactical top-down view on the right. An automatic court detection mode uses HSV colour thresholding to find the court boundary without manual point specification.

### 6.4 Team / Role Clustering
For each confirmed track, the upper 50% of the bounding box (jersey region) is cropped and converted to HSV. An 18-bin hue histogram is extracted and L2-normalised. Per-track histograms are averaged across all frames and fed to `KMeans(k=2)`. The resulting cluster labels correspond to the two teams. Results are visualised in a team-coloured tracking video and a bar chart showing the number of unique IDs per team. The clusterer re-runs every 60 frames as more data accumulates.

### 6.5 Speed Estimation
For each track, consecutive centre-point positions are used to compute frame-to-frame pixel displacements. These are divided by the frame interval and multiplied by the video frame rate to obtain speed in px/s. With a `pixels_per_metre` calibration factor (measured from a known court dimension in the frame), this converts to m/s or km/h. A live speed label is overlaid on each bounding box in the speed video output. A summary CSV records average speed, peak speed, and total distance covered per track.

### 6.6 Object Count Over Time
`main.py` records `(frame_number, confirmed_track_count)` at every processed frame. This is saved as `count_over_time.csv`. `evaluation.py` plots the raw count alongside a 30-frame rolling average and the count distribution histogram, giving a clear picture of how many players were visible throughout the clip.

### 6.7 Evaluation Metrics (No Ground Truth)
Since manual annotations are not provided, proxy metrics are computed from the track data:

| Metric | Description |
|--------|-------------|
| Unique IDs | Total track IDs assigned. Ideal = true player count. Excess = ID switches. |
| Avg track lifetime | Mean frames a track remains active. Longer = more stable. |
| Fragmentation ratio | `(n_ids − modal_count) / n_ids`. Zero = perfect; higher = more re-IDs. |
| Avg detections/frame | Mean number of confirmed tracks per processed frame. |

Track duration is also plotted as a histogram and a rank-ordered bar chart, allowing visual identification of short-lived ghost tracks vs. long-lived stable tracks.

### 6.8 Model Comparison — YOLOv8n vs YOLOv8s
Both models are run on the same 80 sample frames from the input video. Three metrics are compared:
- **Inference time** (ms/frame) — YOLOv8n is consistently 2–3× faster
- **Detection count** — YOLOv8s typically detects 10–20% more persons at the same confidence threshold, catching partially-occluded players that YOLOv8n misses
- **Confidence distribution** — YOLOv8s produces higher-confidence predictions, meaning fewer borderline detections are admitted or rejected at the 0.5 threshold

The trade-off is clear: YOLOv8n is the right choice for CPU-bound real-time processing; YOLOv8s is preferable when accuracy is the priority and GPU or offline processing is available.

---

## 7. Challenges Faced

| Challenge | Observation and Mitigation |
|-----------|---------------------------|
| Audience filtering | Without the height filter, YOLOv8 detected hundreds of audience members, flooding the tracker with irrelevant tentative tracks and inflating ID counts. The 120 px minimum height filter resolved this. |
| Dense play clusters | When 3–4 players cluster tightly (e.g. a basketball screen or football set piece), bounding boxes overlap substantially. DeepSORT sometimes merges two identities for 2–5 frames. Setting `max_cosine_distance=0.2` (stricter than the default 0.4) reduced but did not eliminate this. |
| Same-team jerseys | Players on the same team wearing identical uniforms stress the Re-ID component. The 128-d CNN embedding captures body shape and number patches, but under broadcast zoom, jerseys appear too small for reliable discrimination. |
| Scoreboard / overlay detections | Some frames produced spurious person detections on scoreboard graphics. The aspect-ratio filter (height ≥ 0.5 × width) eliminated most of these. |
| Camera panning | A lateral pan shifts all tracks simultaneously. The Kalman Filter's constant-velocity model does not account for this, briefly producing large prediction errors. No camera-motion compensation was implemented in this version. |

---

## 8. Failure Cases Observed

- **ID switch on crossing paths:** Two players running past each other sometimes exchange IDs for 2–5 frames. The appearance embedding corrects the assignment after a brief period.
- **Ghost track at frame edge:** A player partially visible at the frame boundary can trigger a second low-quality detection. With `n_init=5` this almost always self-resolves without becoming a confirmed track.
- **Long occlusion (> 50 frames):** If a player is fully obscured for more than ~1.5 s, the track expires and a new ID is assigned on reappearance. This is visible as a colour change in the trajectory visualisation.
- **Bird's-eye auto-detection failure:** On broadcast footage with complex lighting, the HSV court-boundary detection occasionally returns a non-quad contour. The fallback (full-frame corners) produces a less accurate but still functional projection.

---

## 9. Possible Improvements

| Improvement | Expected Benefit |
|-------------|-----------------|
| ByteTrack as alternative tracker | Uses low-confidence detections in a second matching pass; improves recall in crowded scenes without increasing ID switches |
| StrongSORT | Adds camera motion compensation and improved Re-ID model; directly addresses the panning camera failure case |
| YOLOv8m or YOLOv8l | Higher mAP; fewer missed detections under heavy occlusion |
| Re-ID model fine-tuning on sports data | Domain-adapted appearance features; better within-team discrimination |
| Homography-based speed calibration | Accurate m/s figures without manual pixel-per-metre measurement |
| MOT ground-truth annotation | Enables MOTA / IDF1 / HOTA computation for rigorous evaluation |
| GPU inference | 5–10× speed-up; enables `frame_skip=1` for maximum tracking continuity |
| Temporal ensemble detection | Average YOLO predictions over 2–3 consecutive frames to stabilise detections under motion blur |

---

## 10. Conclusion

The YOLOv8n + DeepSORT pipeline satisfies all mandatory and optional requirements of the assignment. Detection is robust against audience noise through calibrated size and aspect-ratio filters. Tracking maintains consistent IDs through a combination of Kalman motion prediction, appearance Re-ID, and a generous `max_age` buffer. The extended analytics suite — heatmap, trajectories, speed, team clustering, bird's-eye view, evaluation metrics, and a model comparison — provides a comprehensive view of player activity and demonstrates how the tracking output can be applied to practical sports analysis tasks.
