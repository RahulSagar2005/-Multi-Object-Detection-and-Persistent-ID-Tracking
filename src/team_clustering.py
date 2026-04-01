"""
team_clustering.py
------------------
Clusters tracked players into two teams based on jersey colour.

Algorithm:
  1. For each confirmed track crop the upper-body region (top 50% of bbox).
  2. Convert crop to HSV and compute a histogram over Hue channel.
  3. Reduce each histogram to a compact feature vector.
  4. Run KMeans(k=2) to assign Team A / Team B labels.
  5. Labels are cached per track_id and used by the visualiser.

Usage (standalone):
    python team_clustering.py  ← reads track_data.csv and a reference frame

Usage (integrated into main pipeline):
    from team_clustering import TeamClusterer
    clusterer = TeamClusterer()
    ...
    label = clusterer.update(track_id, frame_crop)
"""

import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ── Team colour palette (BGR) ─────────────────────────────────────────────
TEAM_COLORS = {
    0: (255, 100,  50),   # Team A — orange-ish
    1: ( 50, 100, 255),   # Team B — blue-ish
   -1: (180, 180, 180),   # Unknown / referee
}

TEAM_LABELS = {0: "Team A", 1: "Team B", -1: "?"}


class TeamClusterer:
    """
    Online jersey-colour clusterer.

    Call update() on every confirmed track; call assign_teams() after
    enough samples have been collected (≥ min_samples_to_cluster).
    """

    def __init__(self, n_bins: int = 18, min_samples_to_cluster: int = 30):
        """
        Args:
            n_bins:                  Number of Hue histogram bins.
            min_samples_to_cluster:  Minimum distinct track-ids before clustering.
        """
        self.n_bins   = n_bins
        self.min_samples = min_samples_to_cluster

        self._features: dict[int, list[np.ndarray]] = defaultdict(list)
        self._labels:   dict[int, int]               = {}   # track_id → 0/1/-1
        self._clustered = False

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_feature(self, crop: np.ndarray) -> np.ndarray | None:
        """Return normalised HSV-hue histogram for the crop."""
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [self.n_bins], [0, 180])
        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, track_id: int, frame: np.ndarray,
               x1: int, y1: int, x2: int, y2: int) -> int:
        """
        Collect jersey-colour feature for this track.

        Returns the current team label (−1 = not yet assigned).
        """
        # Use upper 50 % of bounding box (jersey, not shorts)
        mid_y = (y1 + y2) // 2
        crop  = frame[y1:mid_y, x1:x2]

        feat = self._extract_feature(crop)
        if feat is not None:
            self._features[track_id].append(feat)

        return self._labels.get(track_id, -1)

    def assign_teams(self) -> dict[int, int]:
        """
        Run KMeans on all collected features and return {track_id: team_label}.
        Call this periodically (e.g. every 60 frames) after enough samples.
        """
        ids = [tid for tid, feats in self._features.items() if len(feats) >= 3]
        if len(ids) < self.min_samples:
            return self._labels

        X = np.array([
            np.mean(self._features[tid], axis=0)
            for tid in ids
        ])
        X = normalize(X)

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(X)

        for tid, label in zip(ids, km.labels_):
            self._labels[tid] = int(label)

        self._clustered = True
        return self._labels

    def get_label(self, track_id: int) -> int:
        """Return team label for a given track_id (−1 if unknown)."""
        return self._labels.get(track_id, -1)

    def get_color(self, track_id: int) -> tuple:
        return TEAM_COLORS.get(self.get_label(track_id), TEAM_COLORS[-1])

    def get_team_name(self, track_id: int) -> str:
        return TEAM_LABELS.get(self.get_label(track_id), "?")

    @property
    def is_clustered(self) -> bool:
        return self._clustered


# ── Standalone demo ───────────────────────────────────────────────────────

def demo_from_csv(track_csv: str, video_path: str, output_path: str):
    """
    Re-read track_data.csv and original video to produce a
    team-coloured annotated video.
    """
    import csv
    from collections import defaultdict

    # Load track data
    rows = []
    with open(track_csv) as f:
        for row in csv.DictReader(f):
            rows.append({k: int(v) for k, v in row.items()})

    cap = cv2.VideoCapture(video_path)
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (w, h))

    clusterer = TeamClusterer(min_samples_to_cluster=5)

    # Group rows by frame
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        tracks_in_frame = by_frame.get(frame_count, [])

        for r in tracks_in_frame:
            tid = r["track_id"]
            clusterer.update(tid, frame, r["x1"], r["y1"], r["x2"], r["y2"])

        # Re-cluster every 60 frames
        if frame_count % 60 == 0:
            clusterer.assign_teams()

        # Draw
        for r in tracks_in_frame:
            tid   = r["track_id"]
            color = clusterer.get_color(tid)
            label = f"{clusterer.get_team_name(tid)} ID{tid}"
            cv2.rectangle(frame, (r["x1"], r["y1"]), (r["x2"], r["y2"]), color, 2)
            cv2.putText(frame, label, (r["x1"], r["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✅  Team-clustered video → {output_path}")
    print(f"   Team assignments: {clusterer._labels}")


if __name__ == "__main__":
    import sys
    track_csv  = sys.argv[1] if len(sys.argv) > 1 else "../data/output/track_data.csv"
    video_path = sys.argv[2] if len(sys.argv) > 2 else "../data/input/video.mp4"
    out_path   = sys.argv[3] if len(sys.argv) > 3 else "../data/output/team_output.mp4"
    demo_from_csv(track_csv, video_path, out_path)