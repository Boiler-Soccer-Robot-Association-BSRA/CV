"""Tests for detection-team/inference/visualize.py's drawing and YOLO
output conversion. Uses fake ultralytics results, so no torch or weights
are needed."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.classes import BALL, CLASSES  # noqa: E402
from shared.types import BoundingBox, Detection  # noqa: E402

VISUALIZE_PATH = REPO_ROOT / "detection-team" / "inference" / "visualize.py"
spec = importlib.util.spec_from_file_location("visualize", VISUALIZE_PATH)
visualize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize)


def blank_image(height: int = 120, width: int = 160) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def ball(confidence: float = 0.9, bbox: BoundingBox | None = None) -> Detection:
    return Detection(
        class_name=BALL,
        confidence=confidence,
        bbox=bbox or BoundingBox(x_min=40.0, y_min=50.0, x_max=90.0, y_max=100.0),
    )


def test_draw_does_not_modify_input_and_keeps_shape():
    image = blank_image()
    canvas = visualize.draw_detections(image, [ball()])
    assert canvas.shape == image.shape
    assert canvas.dtype == np.uint8
    assert not image.any()
    assert canvas.any()


def test_draw_with_no_detections_returns_unchanged_copy():
    image = blank_image()
    canvas = visualize.draw_detections(image, [])
    assert np.array_equal(canvas, image)
    assert canvas is not image


def test_box_edge_is_drawn_in_class_color():
    canvas = visualize.draw_detections(blank_image(), [ball()])
    # Left edge of the box, below where the label sits.
    assert tuple(int(v) for v in canvas[90, 40]) == visualize.class_color(BALL)


def test_min_confidence_skips_low_confidence_detections():
    canvas = visualize.draw_detections(blank_image(), [ball(confidence=0.3)], min_confidence=0.5)
    assert not canvas.any()


def test_boxes_partly_outside_the_image_do_not_crash():
    off_edge = BoundingBox(x_min=-20.0, y_min=-30.0, x_max=200.0, y_max=60.0)
    canvas = visualize.draw_detections(blank_image(), [ball(bbox=off_edge)])
    assert canvas.any()


def test_class_colors_are_stable_and_distinct_for_our_classes():
    colors = [visualize.class_color(name) for name in CLASSES]
    assert len(set(colors)) == len(CLASSES)
    assert visualize.class_color("sports ball") == visualize.class_color("sports ball")


def fake_box(cls: int, conf: float, xyxy: list[float]) -> SimpleNamespace:
    # Mirrors the tensor shapes ultralytics uses: box.cls[0], box.conf[0], box.xyxy[0].tolist()
    return SimpleNamespace(cls=[cls], conf=[conf], xyxy=[SimpleNamespace(tolist=lambda: xyxy)])


def test_yolo_results_to_detections_keeps_model_class_names():
    result = SimpleNamespace(
        names={0: "person", 32: "sports ball"},
        boxes=[fake_box(32, 0.75, [1.0, 2.0, 3.0, 4.0]), fake_box(0, 0.5, [5.0, 6.0, 7.0, 8.0])],
    )
    detections = visualize.yolo_results_to_detections([result])
    assert [d.class_name for d in detections] == ["sports ball", "person"]
    assert detections[0].confidence == 0.75
    assert detections[0].bbox == BoundingBox(x_min=1.0, y_min=2.0, x_max=3.0, y_max=4.0)


def test_collect_images_expands_folders_and_skips_non_images(tmp_path):
    (tmp_path / "b.png").touch()
    (tmp_path / "a.jpg").touch()
    (tmp_path / "notes.txt").touch()
    found = visualize.collect_images([tmp_path])
    assert [p.name for p in found] == ["a.jpg", "b.png"]
