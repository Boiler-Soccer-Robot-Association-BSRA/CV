"""Shared type definitions passed between subteams.

Detection produces `Detection` objects; integration consumes them and
produces `PerceptionObject` objects (robot-relative positions) that get
published as ROS 2 perception messages. Keeping these dataclasses here
means the message schema in integration-team/messages/ and the model
output format in detection-team/inference/ stay in sync.
"""

from dataclasses import dataclass


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class Detection:
    """Raw output from the detection model, in image space."""

    class_name: str
    confidence: float
    bbox: BoundingBox
    timestamp: float | None = None
    """Unix time the source camera frame was captured, if known."""


@dataclass
class Position2D:
    """Robot-relative position on the ground plane, in meters."""

    x: float
    y: float


@dataclass
class PerceptionObject:
    """A Detection projected into robot-relative world coordinates."""

    class_name: str
    confidence: float
    position: Position2D
    timestamp: float | None = None
    """Carried over from the source Detection."""
