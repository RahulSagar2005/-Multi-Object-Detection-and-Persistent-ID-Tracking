"""
tracker.py
----------
Wraps deep-sort-realtime's DeepSort tracker.

DeepSORT combines:
  - Kalman Filter  → predicts next location of each track
  - Appearance Re-ID (CNN embeddings) → distinguishes similar-looking people
  - Hungarian Algorithm → optimal detection-to-track assignment

Key parameters:
  max_age            : frames a track is kept alive without a matching detection
  n_init             : detections needed to confirm a new track (prevents ghost tracks)
  max_cosine_distance: similarity threshold for appearance matching (lower = stricter)
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(
        self,
        max_age: int = 50,
        n_init: int = 5,
        max_cosine_distance: float = 0.2,
    ):
        """
        Args:
            max_age:              Frames to keep a lost track alive.
                                  Increase for longer occlusions.
            n_init:               Consecutive detections before a track is
                                  confirmed. Higher = fewer false tracks.
            max_cosine_distance:  Max cosine distance for appearance matching.
                                  Lower = stricter re-identification.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
        )

    def update(
        self,
        detections: list,
        frame,
    ) -> list[tuple[int, int, int, int, int]]:
        """
        Update tracks with current-frame detections.

        Args:
            detections: Output of Detector.detect()  →  [(bbox, conf, label), …]
            frame:      Current BGR frame (needed for appearance feature extraction)

        Returns:
            List of confirmed tracks as (x1, y1, x2, y2, track_id).
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            results.append((int(l), int(t), int(r), int(b), track_id))

        return results