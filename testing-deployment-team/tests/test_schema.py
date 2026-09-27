"""Validates the shared detection/perception schema (shared/types.py,
shared/classes.py) that every subteam is expected to produce/consume."""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.classes import CLASSES
from shared.types import BoundingBox, Detection, PerceptionObject, Position2D


def make_bbox() -> BoundingBox:
    return BoundingBox(x_min=1.0, y_min=2.0, x_max=10.0, y_max=20.0)


def test_bounding_box_fields():
    bbox = make_bbox()
    assert dataclasses.fields(BoundingBox)
    assert (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) == (1.0, 2.0, 10.0, 20.0)


def test_detection_has_required_fields_and_types():
    field_names = {f.name for f in dataclasses.fields(Detection)}
    assert field_names == {"class_name", "confidence", "bbox", "timestamp"}

    detection = Detection(class_name="ball", confidence=0.9, bbox=make_bbox(), timestamp=1234.5)
    assert isinstance(detection.class_name, str)
    assert isinstance(detection.confidence, float)
    assert isinstance(detection.bbox, BoundingBox)
    assert isinstance(detection.timestamp, float)


def test_detection_timestamp_is_optional():
    detection = Detection(class_name="ball", confidence=0.9, bbox=make_bbox())
    assert detection.timestamp is None


def test_detection_class_name_should_be_a_known_class():
    detection = Detection(class_name="ball", confidence=0.9, bbox=make_bbox())
    assert detection.class_name in CLASSES


def test_perception_object_has_required_fields():
    field_names = {f.name for f in dataclasses.fields(PerceptionObject)}
    assert field_names == {"class_name", "confidence", "position", "timestamp"}

    obj = PerceptionObject(
        class_name="ball", confidence=0.9, position=Position2D(x=1.0, y=2.0), timestamp=1234.5
    )
    assert isinstance(obj.position, Position2D)
    assert isinstance(obj.timestamp, float)


def test_confidence_is_a_valid_probability():
    detection = Detection(class_name="ball", confidence=0.5, bbox=make_bbox())
    assert 0.0 <= detection.confidence <= 1.0


def test_bounding_box_is_well_formed():
    bbox = make_bbox()
    assert bbox.x_max > bbox.x_min
    assert bbox.y_max > bbox.y_min
