"""
detector.py
-----------
YOLOv8-based person detector with configurable filters.

Filters applied:
  - Class = 0 (person only)
  - Confidence >= conf_threshold
  - Bounding box height >= min_height  (removes far-away audience)
  - Aspect ratio: height >= width * 0.5 (removes horizontal false positives)
"""

from ultralytics import YOLO


class Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        min_height: int = 120,
    ):
        """
        Args:
            model_path:      Path to the YOLOv8 weights file.
            conf_threshold:  Minimum detection confidence (0–1).
            min_height:      Minimum bounding-box height in pixels.
                             Detections shorter than this are discarded
                             (typically far-away audience members).
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_height = min_height

        # COCO class 0 = person
        self.allowed_classes = [0]

    def detect(self, frame) -> list[tuple[list[float], float, str]]:
        """
        Run inference on a single frame.

        Returns:
            List of (bbox, confidence, label) tuples compatible with
            deep-sort-realtime's update_tracks() call.
            bbox format: [x1, y1, x2, y2]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls = int(box.cls[0])

                # Only person class
                if cls not in self.allowed_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                width  = x2 - x1
                height = y2 - y1

                # Size filter – removes tiny far-away detections
                if height < self.min_height:
                    continue

                # Aspect-ratio sanity check (people are taller than wide)
                if height < width * 0.5:
                    continue

                detections.append(([x1, y1, x2, y2], conf, "person"))

        return detections