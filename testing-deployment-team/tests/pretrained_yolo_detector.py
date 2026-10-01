"""A stand-in detector that wraps stock pretrained COCO YOLO weights, so
the eval harness can run on real images before detection-team has a
RoboCup-trained model.

This is the mitigation ARCHITECTURE.md's cross-team handoff #2 calls
for: prove the harness against a real model on real images now, then
point it at detection-team's checkpoint the moment it exists. It runs
the same weights as scripts/quickstart.py.

COCO doesn't know our classes, so only COCO classes with a reasonable
equivalent are kept (see COCO_TO_BSRA) and everything else is dropped.
In practice that means "sports ball" -> ball. Numbers from this
detector are an interim baseline for the harness, not a measure of
what our own model will do.

Same interface as dummy_detector.py and
detection-team/inference/detector.py (`predict(image) -> list[Detection]`,
constructed with a `weights_path`). Needs `ultralytics` (root
requirements.txt); the weights download on first use.

    python testing-deployment-team/tests/eval_harness.py \\
        --detector-file testing-deployment-team/tests/pretrained_yolo_detector.py \\
        --detector-class PretrainedYoloDetector \\
        --manifest path/to/real/manifest.yaml \\
        --no-baseline-check
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from shared.classes import BALL
from shared.types import BoundingBox, Detection

DEFAULT_WEIGHTS = "yolov8n.pt"  # same stock COCO weights as scripts/quickstart.py

COCO_TO_BSRA = {
    "sports ball": BALL,
}


def to_detections(results: Any, timestamp: float) -> list[Detection]:
    """Convert ultralytics results into shared `Detection`s, keeping only
    classes listed in COCO_TO_BSRA."""
    detections = []
    for result in results:
        for box in result.boxes:
            coco_name = result.names[int(box.cls[0])]
            class_name = COCO_TO_BSRA.get(coco_name)
            if class_name is None:
                continue
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


class PretrainedYoloDetector:
    def __init__(self, weights_path: str | Path | None = None) -> None:
        from ultralytics import YOLO  # imported here so the rest of the harness doesn't need it

        self.model = YOLO(str(weights_path or DEFAULT_WEIGHTS))

    def predict(self, image: np.ndarray) -> list[Detection]:
        timestamp = time.time()
        results = self.model.predict(source=image, verbose=False)
        return to_detections(results, timestamp)
