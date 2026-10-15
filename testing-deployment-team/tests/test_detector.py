"""Tests for detection-team/inference/detector.py. The YOLO model is
replaced with a fake, so no weights are downloaded and no inference runs."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.classes import BALL  # noqa: E402
from shared.types import BoundingBox  # noqa: E402

DETECTOR_PATH = REPO_ROOT / "detection-team" / "inference" / "detector.py"
spec = importlib.util.spec_from_file_location("detector", DETECTOR_PATH)
detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_module)


def fake_box(cls: int, conf: float, xyxy: list[float]) -> SimpleNamespace:
    # Mirrors the tensor shapes ultralytics uses: box.cls[0], box.conf[0], box.xyxy[0].tolist()
    return SimpleNamespace(cls=[cls], conf=[conf], xyxy=[SimpleNamespace(tolist=lambda: xyxy)])


COCO_NAMES = {0: "person", 32: "sports ball"}


def fake_results() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            names=COCO_NAMES,
            boxes=[
                fake_box(32, 0.75, [1.0, 2.0, 3.0, 4.0]),
                fake_box(0, 0.5, [5.0, 6.0, 7.0, 8.0]),
            ],
        )
    ]


def test_without_class_map_keeps_model_names():
    detections = detector_module.results_to_detections(fake_results())
    assert [d.class_name for d in detections] == ["sports ball", "person"]
    assert detections[0].confidence == 0.75
    assert detections[0].bbox == BoundingBox(x_min=1.0, y_min=2.0, x_max=3.0, y_max=4.0)
    assert detections[0].timestamp is None


def test_class_map_renames_and_drops_unmapped_classes():
    detections = detector_module.results_to_detections(
        fake_results(), class_map=detector_module.COCO_CLASS_MAP
    )
    assert [d.class_name for d in detections] == [BALL]


def test_timestamp_is_copied_onto_every_detection():
    detections = detector_module.results_to_detections(fake_results(), timestamp=123.5)
    assert {d.timestamp for d in detections} == {123.5}


class FakeYOLO:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.names = {1: "robot", 0: "ball"}
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [SimpleNamespace(names=self.names, boxes=[fake_box(0, 0.9, [10, 10, 20, 20])])]


@pytest.fixture
def fake_yolo(monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)


def test_detector_predict_passes_settings_and_returns_detections(fake_yolo):
    detector = detector_module.Detector("models/ball.pt", conf=0.4, device="cpu")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    detections = detector.predict(frame, timestamp=99.0)

    assert detector.model.weights == "models/ball.pt"
    kwargs = detector.model.predict_kwargs
    assert kwargs["source"] is frame
    assert (kwargs["conf"], kwargs["device"]) == (0.4, "cpu")
    assert [(d.class_name, d.confidence, d.timestamp) for d in detections] == [(BALL, 0.9, 99.0)]


def test_detector_class_names_are_in_class_id_order(fake_yolo):
    assert detector_module.Detector("models/ball.pt").class_names == ["ball", "robot"]


def test_detector_without_weights_fails_clearly():
    with pytest.raises(ValueError, match="weights_path"):
        detector_module.Detector(None)
