"""Tests for integration-team/messages/perception_message.py without ROS.

CI has no ROS 2, so the message classes are stand-ins built by reading the
real .msg files in integration-team/ros_ws/src/my_robot_controller/msg/.
They behave like rosidl's generated Python classes where it matters: only
known field names, floats must be real floats (ints are rejected), float32
values must fit in float32, and nested messages must be the right type.
If a .msg file changes, these tests check the conversion still matches.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.types import BoundingBox, Detection, PerceptionObject, Position2D  # noqa: E402

MSG_DIR = REPO_ROOT / "integration-team" / "ros_ws" / "src" / "my_robot_controller" / "msg"
SAMPLE_DETECTIONS = REPO_ROOT / "detection-team" / "samples" / "ball_v4_detections.json"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conversion = load(
    "perception_message", REPO_ROOT / "integration-team" / "messages" / "perception_message.py"
)
projection = load("projection", REPO_ROOT / "integration-team" / "geometry" / "projection.py")

FLOAT32_MAX = float(np.finfo(np.float32).max)
PRIMITIVES = {"float32": float, "float64": float, "string": str}


def read_msg(path: Path) -> dict[str, str]:
    """Field name -> type, from a .msg file (comments and blank lines skipped)."""
    fields = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            field_type, name = line.split()
            fields[name] = field_type
    return fields


def make_msg_types() -> SimpleNamespace:
    """A stand-in for `my_robot_controller.msg`, one class per .msg file."""
    specs = {path.stem: read_msg(path) for path in sorted(MSG_DIR.glob("*.msg"))}
    classes: dict[str, type] = {}

    def make_class(type_name: str, fields: dict[str, str]) -> type:
        def __init__(self, **kwargs):
            unknown = set(kwargs) - set(fields)
            assert not unknown, f"{type_name} has no field(s) {unknown}"
            for name, field_type in fields.items():
                if field_type in PRIMITIVES:
                    value = kwargs.get(name, PRIMITIVES[field_type]())
                    py_type = PRIMITIVES[field_type]
                    assert type(value) is py_type, (
                        f"{type_name}.{name} must be {py_type.__name__}, got {type(value).__name__}"
                    )
                    if field_type == "float32":
                        assert abs(value) <= FLOAT32_MAX, f"{type_name}.{name} out of float32 range"
                else:
                    value = kwargs.get(name) or classes[field_type]()
                    assert isinstance(value, classes[field_type]), f"{type_name}.{name} wrong type"
                setattr(self, name, value)

        return type(type_name, (), {"__init__": __init__})

    # Build nested types first (Position2D, BoundingBox) so the others can use them.
    for type_name in sorted(specs, key=lambda t: any(v in specs for v in specs[t].values())):
        classes[type_name] = make_class(type_name, specs[type_name])
    return SimpleNamespace(**classes)


MSG = make_msg_types()


def test_msg_files_match_the_shared_dataclasses():
    # The conversion assumes field names line up one to one.
    assert read_msg(MSG_DIR / "PerceptionObject.msg") == {
        "class_name": "string",
        "confidence": "float32",
        "position": "Position2D",
        "timestamp": "float64",
    }
    assert set(read_msg(MSG_DIR / "Position2D.msg")) == {"x", "y"}
    assert set(read_msg(MSG_DIR / "Detection.msg")) == {
        "class_name",
        "confidence",
        "bbox",
        "timestamp",
    }
    assert set(read_msg(MSG_DIR / "BoundingBox.msg")) == {"x_min", "y_min", "x_max", "y_max"}


def test_perception_object_converts_and_round_trips():
    obj = PerceptionObject("ball", 0.87, Position2D(x=1.52, y=-0.1), timestamp=1731200000.25)
    msg = conversion.to_ros_message(obj, MSG)
    assert isinstance(msg, MSG.PerceptionObject)
    assert (msg.class_name, msg.position.x, msg.position.y) == ("ball", 1.52, -0.1)
    assert msg.timestamp == 1731200000.25
    assert conversion.from_ros_message(msg) == obj


def test_none_timestamp_is_sent_as_zero_and_read_back_as_none():
    obj = PerceptionObject("robot", 0.5, Position2D(2.0, 0.3), timestamp=None)
    msg = conversion.to_ros_message(obj, MSG)
    assert msg.timestamp == conversion.UNSET_TIMESTAMP == 0.0
    assert conversion.from_ros_message(msg).timestamp is None


def test_ints_and_numpy_numbers_become_plain_floats():
    # ROS rejects ints in float fields; projection returns numpy-derived values.
    obj = PerceptionObject("goalpost", 1, Position2D(np.float64(3.0), np.float32(-1)), timestamp=5)
    msg = conversion.to_ros_message(obj, MSG)
    assert type(msg.confidence) is float and type(msg.position.y) is float
    assert msg.timestamp == 5.0


def test_detection_converts_and_round_trips():
    det = Detection("ball", 0.9, BoundingBox(148.1, 372.8, 198.6, 423.3), timestamp=None)
    msg = conversion.to_ros_detection(det, MSG)
    assert isinstance(msg.bbox, MSG.BoundingBox)
    assert (msg.bbox.x_min, msg.bbox.y_max, msg.timestamp) == (148.1, 423.3, 0.0)
    assert conversion.from_ros_detection(msg) == det


def test_missing_ros_workspace_gives_a_clear_error():
    obj = PerceptionObject("ball", 0.5, Position2D(1.0, 0.0))
    with pytest.raises(ImportError, match="colcon build"):
        conversion.to_ros_message(obj)


def sample_detections() -> list[tuple[dict, Detection]]:
    data = json.loads(SAMPLE_DETECTIONS.read_text())
    out = []
    for frame in data["frames"]:
        for d in frame["detections"]:
            out.append(
                (
                    frame,
                    Detection(
                        d["class_name"], d["confidence"], BoundingBox(**d["bbox"]), d["timestamp"]
                    ),
                )
            )
    return out


def test_prototype_pipeline_on_real_detections():
    """detection-team's real ball_v4 output -> projection -> ROS messages.

    The camera here is made up (TORSO-21 frames have no calibration), so
    the positions aren't meaningful, but every step runs on real Detector
    output and every message must be valid."""
    pairs = sample_detections()
    assert pairs, "sample file has no detections"
    published = 0
    for frame, det in pairs:
        focal = frame["width"] / 2 / math.tan(math.radians(35))  # 70 degree horizontal FOV
        params = projection.CameraParams(
            camera_matrix=np.array(
                [[focal, 0, frame["width"] / 2], [0, focal, frame["height"] / 2], [0, 0, 1]]
            ),
            distortion=np.zeros(5),
            mount=projection.CameraMount(height_m=0.6, pitch_deg=30.0),
        )
        for obj in projection.project_detections([det], params):
            msg = conversion.to_ros_message(obj, MSG)
            assert math.isfinite(msg.position.x) and msg.position.x > 0
            assert conversion.from_ros_message(msg).class_name == det.class_name
            published += 1
    assert published > 0


def test_stand_in_messages_are_as_strict_as_ros():
    # Guards the tests above: the stand-ins must reject what ROS rejects.
    with pytest.raises(AssertionError, match="must be float"):
        MSG.Position2D(x=1, y=0.0)
    with pytest.raises(AssertionError, match="no field"):
        MSG.Position2D(x=1.0, y=0.0, z=0.0)
    with pytest.raises(AssertionError, match="float32 range"):
        MSG.Position2D(x=1e39, y=0.0)
    with pytest.raises(AssertionError, match="wrong type"):
        MSG.PerceptionObject(position=MSG.BoundingBox())
