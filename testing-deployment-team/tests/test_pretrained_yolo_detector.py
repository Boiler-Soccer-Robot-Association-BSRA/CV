"""Tests for the pretrained-YOLO stand-in's output conversion.

These use fake ultralytics result objects, so they don't need
ultralytics, torch, or downloaded weights to run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

from pretrained_yolo_detector import to_detections

from shared.classes import BALL
from shared.types import BoundingBox


class FakeTensor:
    """Just enough of a torch tensor for to_detections: indexing and tolist()."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        value = self.values[idx]
        return FakeTensor(value) if isinstance(value, list) else value

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = FakeTensor([cls])
        self.conf = FakeTensor([conf])
        self.xyxy = FakeTensor([xyxy])


class FakeResult:
    names = {0: "person", 32: "sports ball"}

    def __init__(self, boxes):
        self.boxes = boxes


def test_sports_ball_maps_to_ball():
    results = [FakeResult([FakeBox(32, 0.75, [10.0, 20.0, 30.0, 40.0])])]
    detections = to_detections(results, timestamp=123.0)
    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == BALL
    assert det.confidence == 0.75
    assert det.bbox == BoundingBox(10.0, 20.0, 30.0, 40.0)
    assert det.timestamp == 123.0


def test_unmapped_coco_classes_are_dropped():
    results = [
        FakeResult(
            [
                FakeBox(0, 0.9, [0.0, 0.0, 5.0, 5.0]),
                FakeBox(32, 0.6, [1.0, 1.0, 2.0, 2.0]),
            ]
        )
    ]
    detections = to_detections(results, timestamp=0.0)
    assert [d.class_name for d in detections] == [BALL]


def test_no_boxes_gives_no_detections():
    assert to_detections([FakeResult([])], timestamp=0.0) == []
