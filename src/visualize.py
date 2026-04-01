"""
visualize.py
------------
Drawing utilities for the tracking pipeline.

Features:
  - Bounding boxes with unique per-ID colour
  - ID label with filled background for readability
  - Trajectory trails (last N centre-points per ID) — properly faded, not black
  - Frame-level person count overlay
  - Heatmap accumulation helper
"""

import cv2
import numpy as np
import random
from collections import defaultdict, deque

# ── Per-ID colour map ────────────────────────────────────────────────────────
_id_colors: dict = {}

def get_color(track_id: int) -> tuple:
    """Return a consistent bright BGR colour for a given track ID."""
    if track_id not in _id_colors:
        random.seed(track_id * 137)
        r = random.randint(80, 255)
        g = random.randint(80, 255)
        b = random.randint(80, 255)
        ch = [r, g, b]
        ch[ch.index(max(ch))] = 255
        _id_colors[track_id] = tuple(ch)
    return _id_colors[track_id]


# ── Trajectory history ───────────────────────────────────────────────────────
_history: dict = defaultdict(lambda: deque(maxlen=60))


# ── Main drawing function ────────────────────────────────────────────────────
def draw_tracks(
    frame,
    tracks: list,
    trail_length: int = 40,
    draw_trail: bool = True,
):
    for x1, y1, x2, y2, track_id in tracks:
        color = get_color(track_id)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        _history[track_id].append((cx, cy))

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID label with filled background
        label = f"ID {track_id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 10, th + 4)
        cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Trajectory trail — alpha from 0.15 to 1.0, min brightness 40
        if draw_trail:
            pts = list(_history[track_id])[-trail_length:]
            for i in range(1, len(pts)):
                alpha = 0.15 + 0.85 * (i / len(pts))
                faded = tuple(max(40, int(c * alpha)) for c in color)
                thickness = 1 if alpha < 0.5 else 2
                cv2.line(frame, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)

    # Person count overlay
    count_label = f"People: {len(tracks)}"
    (cw, ch2), _ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (8, 8), (cw + 20, 42), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, count_label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


# ── Heatmap helpers ──────────────────────────────────────────────────────────
class HeatmapAccumulator:
    """Accumulates foot-point positions across frames and renders a heatmap."""

    def __init__(self, frame_h: int, frame_w: int):
        self.canvas = np.zeros((frame_h, frame_w), dtype=np.float32)

    def update(self, tracks: list):
        for x1, y1, x2, y2, _ in tracks:
            fx = (x1 + x2) // 2
            fy = y2
            if 0 <= fy < self.canvas.shape[0] and 0 <= fx < self.canvas.shape[1]:
                cv2.circle(self.canvas, (fx, fy), 15, 1.0, -1)

    def render(self, background=None, alpha: float = 0.6):
        normalized = cv2.normalize(self.canvas, None, 0, 255, cv2.NORM_MINMAX)
        heat_u8    = normalized.astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        if background is None:
            return heat_color
        bg   = background.copy()
        mask = heat_u8 > 0
        bg[mask] = cv2.addWeighted(bg, 1 - alpha, heat_color, alpha, 0)[mask]
        return bg


def reset_history():
    _history.clear()
    _id_colors.clear()