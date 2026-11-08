"""export_detections.py turns Detections into JSON with exactly the Detection fields."""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.types import BoundingBox, Detection  # noqa: E402

PATH = REPO_ROOT / "detection-team" / "inference" / "export_detections.py"
spec = importlib.util.spec_from_file_location("export_detections", PATH)
export_detections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_detections)


def test_frame_record_has_detection_fields_and_round_trips_through_json():
    det = Detection("ball", 0.87654, BoundingBox(10.04, 20.0, 30.06, 40.0), timestamp=None)
    record = export_detections.frame_record("a.png", 640, 480, [det])
    back = json.loads(json.dumps(record))
    assert back["image"] == "a.png" and (back["width"], back["height"]) == (640, 480)
    (d,) = back["detections"]
    assert set(d) == {"class_name", "confidence", "bbox", "timestamp"}
    assert set(d["bbox"]) == {"x_min", "y_min", "x_max", "y_max"}
    assert d["confidence"] == 0.877 and d["bbox"]["x_min"] == 10.0 and d["timestamp"] is None
    rebuilt = Detection(d["class_name"], d["confidence"], BoundingBox(**d["bbox"]), d["timestamp"])
    assert rebuilt.bbox.x_max == 30.1


def test_committed_sample_matches_the_format():
    sample = REPO_ROOT / "detection-team" / "samples" / "ball_v4_detections.json"
    data = json.loads(sample.read_text())
    assert {"model", "conf", "frames"} <= set(data)
    for frame in data["frames"]:
        for d in frame["detections"]:
            box = d["bbox"]
            assert 0 <= box["x_min"] < box["x_max"] <= frame["width"]
            assert 0 <= box["y_min"] < box["y_max"] <= frame["height"]
