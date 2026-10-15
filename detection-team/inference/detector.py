"""Reusable inference module: image in, `Detection`s out.

This is what integration-team's ROS 2 node calls, and what
testing-deployment-team benchmarks — keep it notebook-free so both can
import it directly.

    from detector import Detector   # with detection-team/inference/ on sys.path

    detector = Detector("detection-team/models/ball_v1.pt")
    detections = detector.predict(frame)               # frame: BGR uint8 (H, W, 3)
    detections = detector.predict(frame, timestamp=t)  # t: when the frame was captured

A model trained with training/train.py already uses our class names
(shared/classes.py), so no mapping is needed. To run a model trained on
some other label set (e.g. stock COCO weights), pass `class_map` to
rename the classes you care about and drop the rest:

    Detector("yolov8n.pt", class_map=COCO_CLASS_MAP)   # COCO "sports ball" -> "ball"

Same interface as testing-deployment-team/tests/dummy_detector.py, so the
eval harness can load it directly with --detector-file/--detector-class.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.classes import BALL  # noqa: E402
from shared.types import BoundingBox, Detection  # noqa: E402

DEFAULT_CONFIDENCE = 0.25

# COCO classes that have a reasonable equivalent in shared/classes.py.
COCO_CLASS_MAP = {"sports ball": BALL}


def results_to_detections(
    results: Any, class_map: dict[str, str] | None = None, timestamp: float | None = None
) -> list[Detection]:
    """Convert ultralytics `Results` into shared `Detection`s.

    With no `class_map`, class names are kept exactly as the model
    reports them. With one, names are renamed through it and anything
    not in it is dropped.
    """
    detections = []
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            if class_map is not None:
                if class_name not in class_map:
                    continue
                class_name = class_map[class_name]
            x_min, y_min, x_max, y_max = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    timestamp=timestamp,
                )
            )
    return detections


class Detector:
    def __init__(
        self,
        weights_path: str | Path | None,
        conf: float = DEFAULT_CONFIDENCE,
        class_map: dict[str, str] | None = None,
        device: str | None = None,
    ) -> None:
        """
        weights_path: YOLO weights (.pt). A bare name like "yolov8n.pt"
            downloads the stock pretrained weights on first use.
        conf: minimum confidence for a detection to be returned.
        class_map: rename model class names to ours, dropping the rest
            (see module docstring). None keeps the model's names.
        device: "cpu", "mps", "cuda:0", ... None lets ultralytics pick.
        """
        if weights_path is None:
            raise ValueError("Detector needs weights_path (e.g. detection-team/models/ball_v1.pt)")
        from ultralytics import YOLO  # imported here so importing this module stays cheap

        self.weights_path = Path(weights_path)
        self.conf = conf
        self.class_map = class_map
        self.device = device
        self.model = YOLO(str(weights_path))

    @property
    def class_names(self) -> list[str]:
        """The class names the loaded model predicts, in class-id order."""
        names = self.model.names
        return [names[i] for i in sorted(names)]

    def predict(self, image: np.ndarray, timestamp: float | None = None) -> list[Detection]:
        """Run the model on one BGR uint8 frame (see ARCHITECTURE.md's
        camera-frame contract). `timestamp` is the frame's capture time,
        copied onto every Detection; leave it None for static images."""
        results = self.model.predict(
            source=image, conf=self.conf, device=self.device, verbose=False
        )
        return results_to_detections(results, self.class_map, timestamp)
