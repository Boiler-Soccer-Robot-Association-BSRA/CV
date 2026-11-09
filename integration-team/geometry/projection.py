"""Ground-plane projection: image-space detections to robot-relative positions.

This is step 3 of the pipeline in ARCHITECTURE.md. It takes a `Detection`
(a box in pixels, from detection-team's `Detector`) and returns a
`PerceptionObject` (a position on the floor in meters, relative to the
robot), which integration-team then publishes over ROS 2.

Usage
-----
    from projection import CameraMount, load_camera_params, project_detections

    params = load_camera_params("camera_params.yaml")       # calibrate.py output + mount:
    objects = project_detections(detections, params)        # list[PerceptionObject]
    objects[0].position.x, objects[0].position.y            # meters ahead / left

`detection-team/inference/detector.py` explains why modules in the hyphenated
team folders are imported by putting their folder on `sys.path`.

Inputs
------
Projection needs two things about the camera:

1. **Intrinsics + distortion** (`camera_matrix`, `distortion`): how the lens
   maps directions to pixels. Measured by `calibration/calibrate.py` from
   checkerboard photos. Valid for one camera, lens, focus and resolution.
2. **Mount** (`CameraMount`): where the camera sits on the robot and how it
   is tilted. Measured by hand (tape measure, inclinometer), stored in the
   same YAML under `mount:`. See calibration/CALIBRATION_GUIDE.md, Step 6.

How it works
------------
For each detection:

1. `contact_pixel` picks the pixel that touches the floor (bottom-center
   of a robot's box; the center of a ball, one radius above the floor).
2. `cv2.undistortPoints` removes lens distortion, giving a direction in the
   camera's optical frame.
3. `camera_to_robot_rotation` rotates that direction into the robot frame
   using the mount's pitch, roll and yaw.
4. `pixel_to_ground` follows the ray down from the lens until it reaches the
   floor (or the ball-center plane) and returns that point's x and y.

`ground_to_pixel` runs the same math backwards: given a point on the floor,
where does it appear in the image? Tests and validation use it to check
the forward direction.

Coordinate conventions (ROS REP 103; also documented in ARCHITECTURE.md)
------------------------------------------------------------------------
* **Robot frame**: origin on the floor under the robot's reference point;
  **+x forward, +y left, +z up**; meters. A ball 2 m ahead and 0.3 m to the
  robot's left is `Position2D(x=2.0, y=0.3)`; one on the right has y < 0.
* **Camera optical frame** (OpenCV's): +x right in the image, +y down in the
  image, +z out of the lens.
* **Mount angles**, in degrees:
  `pitch_deg` > 0 tilts the camera down toward the floor;
  `yaw_deg` > 0 turns it to the robot's left;
  `roll_deg` > 0 drops the camera's right side (clockwise seen from behind).

If the numbers look wrong
-------------------------
| Symptom                                       | Likely cause                                   |
|-----------------------------------------------|------------------------------------------------|
| Every detection raises ProjectionError        | pitch has the wrong sign (camera "looks up")   |
| Left/right swapped (y has the wrong sign)     | mount yaw/roll sign, or image flipped by driver|
| Distances consistently too far / too near     | pitch too small / too large, or wrong height   |
| Error grows toward the image edges            | bad or missing distortion (re-calibrate)       |
| Balls too far by a few cm, robots fine        | BALL_RADIUS_M wrong for the ball in use        |
| Everything off after changing resolution      | calibration is for a different resolution      |

Pitch and height dominate: with the camera 0.5 m up and tilted 20 degrees,
being 1 degree and 2 cm off moves a point at 2 m by about 6 cm.

Known limits
------------
* The mount is fixed. A walking robot's head moves; when head angles are
  available, build a per-frame mount with `with_mount`.
* Only OpenCV's standard distortion model (what calibrate.py fits). If the
  wide lens needs `cv2.fisheye`, this module and calibrate.py change together.
* The floor is assumed flat, and the robot is assumed upright (no body tilt).
* Field lines are not boxes and are not handled here (team decision, Oct 15).
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.classes import BALL, LANDMARK  # noqa: E402
from shared.types import Detection, PerceptionObject, Position2D  # noqa: E402

BALL_RADIUS_M = 0.095
"""Radius of the competition ball: a size 3 ball (58-61 cm around, so about
19 cm across). A ball resting on the floor has its center this high up, which
is the plane its box center is projected onto. Override per call with
`ball_radius_m` if a different ball is used."""

# A camera with zero pitch/roll/yaw looks straight ahead along robot +x.
# Its optical axes, written as columns in the robot frame, are then:
#   optical +x (image right) = robot -y (right)
#   optical +y (image down)  = robot -z (down)
#   optical +z (lens axis)   = robot +x (forward)
_OPTICAL_TO_ROBOT_LEVEL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# A ray must point at least this far downward (as a unit-vector z component)
# to be treated as hitting the floor. Rays this close to horizontal would
# land kilometers away and are better reported as unprojectable.
_MIN_DOWNWARD = 1e-9


class ProjectionError(ValueError):
    """The pixel can't be placed on the floor: it is at or above the horizon,
    or the point is behind the camera. Callers handling a whole frame should
    skip that detection (see `project_detections`)."""


@dataclass
class CameraMount:
    """Where the camera sits on the robot. Measured by hand, not calibrated.

    All distances are in meters, in the robot frame; angles are in degrees.
    See the module docstring for the sign of each angle.
    """

    height_m: float
    """Height of the lens above the floor."""
    pitch_deg: float = 0.0
    """Downward tilt; positive looks toward the floor."""
    roll_deg: float = 0.0
    """Sideways roll; positive drops the camera's right side."""
    yaw_deg: float = 0.0
    """Left/right turn; positive looks to the robot's left."""
    x_m: float = 0.0
    """How far the lens is ahead of the robot's origin."""
    y_m: float = 0.0
    """How far the lens is left of the robot's origin."""


@dataclass
class CameraParams:
    """Everything projection needs: calibrate.py's output plus the mount."""

    camera_matrix: np.ndarray
    """3x3 intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], in pixels."""
    distortion: np.ndarray
    """OpenCV distortion coefficients (k1, k2, p1, p2, k3, ...)."""
    mount: CameraMount
    resolution: tuple[int, int] | None = None
    """(width, height) the calibration was made at. Frames at another
    resolution give wrong positions; this is recorded so callers can check."""


def load_camera_params(path: str | Path, mount: CameraMount | None = None) -> CameraParams:
    """Read a calibrate.py YAML into `CameraParams`.

    The mount comes from the file's `mount:` section, or from the `mount`
    argument, which wins if both are given (handy for trying different
    tilts without editing the file). Raises ValueError with the missing
    piece named if the file is incomplete.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("camera_matrix", "distortion"):
        if key not in data:
            raise ValueError(f"{path} has no '{key}'; is it calibrate.py's output?")
    if mount is None:
        if "mount" not in data:
            raise ValueError(f"{path} has no 'mount:' section; pass mount=CameraMount(...)")
        mount = CameraMount(**data["mount"])

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"{path}: camera_matrix must be 3x3, got {camera_matrix.shape}")
    res = data.get("resolution")
    return CameraParams(
        camera_matrix=camera_matrix,
        distortion=np.array(data["distortion"], dtype=np.float64),
        mount=mount,
        resolution=(res["width"], res["height"]) if res else None,
    )


def camera_to_robot_rotation(mount: CameraMount) -> np.ndarray:
    """3x3 rotation taking optical-frame directions to robot-frame directions.

    Built as: start level (`_OPTICAL_TO_ROBOT_LEVEL`), roll about the lens
    axis, tilt down about the robot's left (y) axis, then yaw about up (z).
    Its transpose is the inverse (robot to optical).
    """
    pitch, roll, yaw = (np.radians(a) for a in (mount.pitch_deg, mount.roll_deg, mount.yaw_deg))
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # Rotation about +y by +pitch turns forward (+x) toward down (-z).
    tilt = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    # Rotation about +x by +roll turns right (-y) toward down (-z).
    roll_m = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    # Rotation about +z by +yaw turns forward (+x) toward left (+y).
    yaw_m = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return yaw_m @ tilt @ roll_m @ _OPTICAL_TO_ROBOT_LEVEL


def _ray_in_robot_frame(u: float, v: float, params: CameraParams) -> np.ndarray:
    """Unit viewing ray through pixel (u, v), in the robot frame."""
    pts = np.array([[[u, v]]], dtype=np.float64)
    # undistortPoints returns normalized coordinates: the ray (nx, ny, 1) in
    # the optical frame, with lens distortion already removed.
    nx, ny = cv2.undistortPoints(pts, params.camera_matrix, params.distortion)[0, 0]
    ray = camera_to_robot_rotation(params.mount) @ np.array([nx, ny, 1.0])
    return ray / np.linalg.norm(ray)


def pixel_to_ground(
    u: float, v: float, params: CameraParams, plane_height_m: float = 0.0
) -> Position2D:
    """Where the ray through pixel (u, v) meets a horizontal plane
    `plane_height_m` above the floor, as a robot-frame position.

    Raises ProjectionError if the ray never reaches that plane: the pixel
    is at or above the horizon, or the lens is not above the plane.
    """
    m = params.mount
    ray = _ray_in_robot_frame(u, v, params)
    drop = m.height_m - plane_height_m  # vertical distance from lens down to the plane
    if ray[2] > -_MIN_DOWNWARD or drop <= 0:
        raise ProjectionError(
            f"pixel ({u:.0f}, {v:.0f}) does not reach the plane {plane_height_m:.3f} m up "
            f"(camera {m.height_m:.3f} m up, pitch {m.pitch_deg:.1f} deg)"
        )
    distance_along_ray = drop / -ray[2]
    return Position2D(
        x=float(m.x_m + distance_along_ray * ray[0]),
        y=float(m.y_m + distance_along_ray * ray[1]),
    )


def ground_to_pixel(
    x: float, y: float, params: CameraParams, plane_height_m: float = 0.0
) -> tuple[float, float]:
    """Inverse of `pixel_to_ground`: the pixel (u, v) where the robot-frame
    point (x, y, plane_height_m) appears, including lens distortion.

    Used by the tests and by accuracy validation to place known points.
    Raises ProjectionError if the point is behind the camera.
    """
    m = params.mount
    offset = np.array([x - m.x_m, y - m.y_m, plane_height_m - m.height_m])
    in_optical = camera_to_robot_rotation(m).T @ offset
    if in_optical[2] <= 0:
        raise ProjectionError(f"point ({x:.2f}, {y:.2f}) is behind the camera")
    no_rotation = no_translation = np.zeros(3)  # point is already in the camera frame
    px, _ = cv2.projectPoints(
        in_optical.reshape(1, 1, 3),
        no_rotation,
        no_translation,
        params.camera_matrix,
        params.distortion,
    )
    return float(px[0, 0, 0]), float(px[0, 0, 1])


def contact_pixel(
    detection: Detection, ball_radius_m: float = BALL_RADIUS_M
) -> tuple[float, float, float]:
    """The pixel to project for a detection, and the height of the plane to
    project it onto: (u, v, plane_height_m).

    * ball: box center, onto a plane one ball radius up. The bottom edge of
      a ball's box is not where it touches the floor in the image (the ball
      bulges toward the camera), but its center is always one radius up.
    * landmark: box center, on the floor. data-team draws these boxes as
      squares around a labeled point on the floor.
    * everything else (robot, goalpost): bottom-center of the box, on the
      floor, where it stands. Wrong if the bottom is hidden or cut off.
    """
    box = detection.bbox
    u = (box.x_min + box.x_max) / 2
    v_center = (box.y_min + box.y_max) / 2
    if detection.class_name == BALL:
        return u, v_center, ball_radius_m
    if detection.class_name == LANDMARK:
        return u, v_center, 0.0
    return u, box.y_max, 0.0


def project_to_ground_plane(
    detection: Detection, camera_params: CameraParams, ball_radius_m: float = BALL_RADIUS_M
) -> PerceptionObject:
    """Project one detection to a robot-relative position, in meters.

    class_name, confidence and timestamp are copied unchanged. Raises
    ProjectionError if the detection's floor-contact point is at or above
    the horizon, e.g. a box cut off at the top of the frame.
    """
    u, v, plane_height = contact_pixel(detection, ball_radius_m)
    position = pixel_to_ground(u, v, camera_params, plane_height)
    return PerceptionObject(
        class_name=detection.class_name,
        confidence=detection.confidence,
        position=position,
        timestamp=detection.timestamp,
    )


def project_detections(
    detections: list[Detection], camera_params: CameraParams, ball_radius_m: float = BALL_RADIUS_M
) -> list[PerceptionObject]:
    """`project_to_ground_plane` over one frame's detections, silently
    skipping any that can't be placed on the floor. Output order follows
    the input order, minus the skipped ones."""
    objects = []
    for detection in detections:
        try:
            objects.append(project_to_ground_plane(detection, camera_params, ball_radius_m))
        except ProjectionError:
            continue
    return objects


def with_mount(params: CameraParams, **changes: float) -> CameraParams:
    """A copy of `params` with some mount fields changed, leaving the
    original untouched. E.g. `with_mount(params, pitch_deg=head_pitch)` to
    use this frame's head angle."""
    return replace(params, mount=replace(params.mount, **changes))
