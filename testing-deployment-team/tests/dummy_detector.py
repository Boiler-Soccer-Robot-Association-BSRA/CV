"""A detector that finds the bright ball in fixtures/sample_dataset via
simple color thresholding — no model, no weights.

It exists so the benchmark harness has something real to run end to
end today. It implements the same interface as
`detection-team/inference/detector.py`'s `Detector`
(`predict(image) -> list[Detection]`, constructed with a `weights_path`
it's free to ignore), so eval_harness.py can point at either one
interchangeably. Swap in the real one once it's implemented:

    python testing-deployment-team/tests/eval_harness.py \\
        --detector-file detection-team/inference/detector.py \\
        --detector-class Detector \\
        --weights path/to/weights.pt \\
        --manifest path/to/real/manifest.yaml
"""

import time
from pathlib import Path

import cv2
import numpy as np

from shared.classes import BALL
from shared.types import BoundingBox, Detection

BRIGHTNESS_THRESHOLD = 400  # sum of B+G+R channel values
MIN_AREA = 20


class DummyDetector:
    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = weights_path  # unused; kept for interface parity

    def predict(self, image: np.ndarray) -> list[Detection]:
        timestamp = time.time()
        brightness = image.astype(np.int32).sum(axis=2)
        mask = (brightness > BRIGHTNESS_THRESHOLD).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(
                x_min=float(x), y_min=float(y), x_max=float(x + w), y_max=float(y + h)
            )
            detections.append(
                Detection(class_name=BALL, confidence=0.85, bbox=bbox, timestamp=timestamp)
            )
        return detections
