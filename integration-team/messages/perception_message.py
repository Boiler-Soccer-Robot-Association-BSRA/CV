"""Convert between shared/types.py dataclasses and the ROS 2 messages.

The messages are defined in
integration-team/ros_ws/src/my_robot_controller/msg/ (#33, #37), one
`.msg` per dataclass with the same field names:

| Python (shared/types.py) | ROS 2 (my_robot_controller/msg) |
|---|---|
| `PerceptionObject` | `PerceptionObject`: class_name, confidence, position, timestamp |
| `Position2D` | `Position2D`: float32 x, y (meters; +x forward, +y left) |
| `Detection` | `Detection`: class_name, confidence, bbox, timestamp |
| `BoundingBox` | `BoundingBox`: float32 x_min, y_min, x_max, y_max (pixels) |

    msg = to_ros_message(perception_object)       # publish this
    obj = from_ros_message(msg)                   # in a subscriber

**Unset timestamps.** A ROS field can't be `None`, so a `timestamp` of
`None` (no known capture time, e.g. a static test image) is sent as
`UNSET_TIMESTAMP` (0.0) and read back as `None`. Real capture times are
Unix seconds, so 0.0 (1 Jan 1970) never collides with one. *Proposed by
integration-team; ARCHITECTURE.md lists this as an open team question.*

**Testing without ROS.** The message classes only exist inside a built
and sourced ROS 2 workspace (`colcon build`, then `source
install/setup.bash`). Every function takes an optional `msg_types`, any
object with `PerceptionObject`, `Position2D`, `Detection` and
`BoundingBox` attributes, so tests (and CI, which has no ROS) can pass
stand-ins. Leave it out on the robot to use `my_robot_controller.msg`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.types import BoundingBox, Detection, PerceptionObject, Position2D  # noqa: E402

UNSET_TIMESTAMP = 0.0
"""What a `None` timestamp becomes in a message (ROS fields can't be None)."""


def _default_msg_types():
    """The generated ROS 2 message module. Only importable inside a built,
    sourced ROS 2 workspace; see the module docstring."""
    try:
        from my_robot_controller import msg
    except ImportError as e:
        raise ImportError(
            "my_robot_controller.msg not found: build the ROS 2 workspace "
            "(integration-team/ros_ws: colcon build) and source install/setup.bash, "
            "or pass msg_types= for testing"
        ) from e
    return msg


def _timestamp_to_ros(timestamp: float | None) -> float:
    return UNSET_TIMESTAMP if timestamp is None else float(timestamp)


def _timestamp_from_ros(timestamp: float) -> float | None:
    return None if timestamp == UNSET_TIMESTAMP else float(timestamp)


def to_ros_message(obj: PerceptionObject, msg_types=None):
    """A `PerceptionObject` as a `my_robot_controller/msg/PerceptionObject`.

    Numbers are converted to plain Python floats, since ROS message fields
    reject ints and some numpy types. Positions are float32 in the message,
    so they keep about 7 significant digits (well under a millimeter).
    """
    types = msg_types or _default_msg_types()
    return types.PerceptionObject(
        class_name=str(obj.class_name),
        confidence=float(obj.confidence),
        position=types.Position2D(x=float(obj.position.x), y=float(obj.position.y)),
        timestamp=_timestamp_to_ros(obj.timestamp),
    )


def from_ros_message(msg) -> PerceptionObject:
    """A received `PerceptionObject` message back as the shared dataclass."""
    return PerceptionObject(
        class_name=msg.class_name,
        confidence=float(msg.confidence),
        position=Position2D(x=float(msg.position.x), y=float(msg.position.y)),
        timestamp=_timestamp_from_ros(msg.timestamp),
    )


def to_ros_detection(detection: Detection, msg_types=None):
    """A `Detection` as a `my_robot_controller/msg/Detection`, for nodes
    that publish raw image-space detections before projection."""
    types = msg_types or _default_msg_types()
    box = detection.bbox
    return types.Detection(
        class_name=str(detection.class_name),
        confidence=float(detection.confidence),
        bbox=types.BoundingBox(
            x_min=float(box.x_min),
            y_min=float(box.y_min),
            x_max=float(box.x_max),
            y_max=float(box.y_max),
        ),
        timestamp=_timestamp_to_ros(detection.timestamp),
    )


def from_ros_detection(msg) -> Detection:
    """A received `Detection` message back as the shared dataclass."""
    box = msg.bbox
    return Detection(
        class_name=msg.class_name,
        confidence=float(msg.confidence),
        bbox=BoundingBox(
            x_min=float(box.x_min),
            y_min=float(box.y_min),
            x_max=float(box.x_max),
            y_max=float(box.y_max),
        ),
        timestamp=_timestamp_from_ros(msg.timestamp),
    )
