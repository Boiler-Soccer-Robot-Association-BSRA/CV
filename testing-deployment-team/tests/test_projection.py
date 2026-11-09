"""Tests for integration-team/geometry/projection.py using a synthetic camera
with a known pose, so no photos or hardware are needed."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.classes import BALL, GOALPOST, LANDMARK, ROBOT  # noqa: E402
from shared.types import BoundingBox, Detection  # noqa: E402

PROJECTION_PATH = REPO_ROOT / "integration-team" / "geometry" / "projection.py"
spec = importlib.util.spec_from_file_location("projection", PROJECTION_PATH)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)

CameraMount = projection.CameraMount
CameraParams = projection.CameraParams
ProjectionError = projection.ProjectionError

WIDTH, HEIGHT = 1920, 1200
FOCAL = 700.0
CAMERA_MATRIX = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]])
CX, CY = WIDTH / 2, HEIGHT / 2


def make_params(distortion=None, **mount) -> CameraParams:
    mount = {"height_m": 0.5, "pitch_deg": 20.0, **mount}
    return CameraParams(
        camera_matrix=CAMERA_MATRIX,
        distortion=np.array(distortion if distortion is not None else [0.0] * 5),
        mount=CameraMount(**mount),
        resolution=(WIDTH, HEIGHT),
    )


def det(class_name, x_min, y_min, x_max, y_max, timestamp=None) -> Detection:
    return Detection(class_name, 0.9, BoundingBox(x_min, y_min, x_max, y_max), timestamp)


def test_image_center_hits_floor_where_geometry_says():
    # Lens 0.5 m up, tilted 20 deg down: the axis meets the floor 0.5/tan(20 deg) ahead.
    pos = projection.pixel_to_ground(CX, CY, make_params())
    assert pos.x == pytest.approx(0.5 / math.tan(math.radians(20)), abs=1e-6)
    assert pos.y == pytest.approx(0.0, abs=1e-6)


def test_lower_in_image_is_closer_and_left_in_image_is_left():
    params = make_params()
    centre = projection.pixel_to_ground(CX, CY, params)
    lower = projection.pixel_to_ground(CX, CY + 200, params)
    left = projection.pixel_to_ground(CX - 200, CY + 100, params)
    right = projection.pixel_to_ground(CX + 200, CY + 100, params)
    assert lower.x < centre.x
    assert left.y > 0 > right.y  # +y is left
    assert left.y == pytest.approx(-right.y, abs=1e-6)


@pytest.mark.parametrize("distortion", [None, [-0.3, 0.1, 0.0, 0.0, -0.02]])
@pytest.mark.parametrize(
    "mount",
    [
        {},
        {"pitch_deg": 35.0, "height_m": 0.4},
        {"roll_deg": 5.0, "yaw_deg": -10.0},
        {"x_m": 0.06, "y_m": -0.03, "height_m": 0.55},
    ],
)
def test_ground_point_survives_a_round_trip_through_the_image(distortion, mount):
    params = make_params(distortion, **mount)
    for x, y in [(1.0, 0.0), (2.0, 0.5), (2.0, -0.7), (3.5, 1.2), (0.9, -0.2)]:
        u, v = projection.ground_to_pixel(x, y, params)
        pos = projection.pixel_to_ground(u, v, params)
        assert (pos.x, pos.y) == pytest.approx((x, y), abs=2e-3)


def test_pixel_at_or_above_horizon_is_rejected():
    params = make_params(pitch_deg=0.0)
    with pytest.raises(ProjectionError):
        projection.pixel_to_ground(CX, CY, params)  # straight at the horizon
    with pytest.raises(ProjectionError):
        projection.pixel_to_ground(CX, 10, params)  # above it
    assert projection.pixel_to_ground(CX, CY + 50, params).x > 0


def test_yaw_turns_the_view_left_and_roll_changes_the_side_seen():
    ahead = projection.pixel_to_ground(CX, CY, make_params())
    turned = projection.pixel_to_ground(CX, CY, make_params(yaw_deg=30.0))
    assert turned.y > 0.1 and turned.x < ahead.x
    # Rolling the camera's right side down swings its "down" toward the robot's left,
    # so a pixel below the image centre now sees ground slightly to the left.
    rolled = projection.pixel_to_ground(CX, CY + 100, make_params(roll_deg=10.0))
    level = projection.pixel_to_ground(CX, CY + 100, make_params())
    assert rolled.y > level.y + 0.01


def test_mount_offset_shifts_every_position():
    base = projection.pixel_to_ground(CX, CY + 80, make_params())
    moved = projection.pixel_to_ground(CX, CY + 80, make_params(x_m=0.1, y_m=0.05))
    assert moved.x == pytest.approx(base.x + 0.1)
    assert moved.y == pytest.approx(base.y + 0.05)


def test_ball_uses_its_centre_and_a_raised_plane():
    params = make_params()
    radius = 0.065
    # A ball resting 1.5 m ahead: its centre is `radius` above the floor.
    u, v = projection.ground_to_pixel(1.5, 0.2, params, plane_height_m=radius)
    half = 12
    ball = det(BALL, u - half, v - half, u + half, v + half)
    obj = projection.project_to_ground_plane(ball, params, ball_radius_m=radius)
    assert (obj.position.x, obj.position.y) == pytest.approx((1.5, 0.2), abs=2e-3)
    # Treating the centre as if it were on the floor would put it too far away.
    naive = projection.pixel_to_ground(u, v, params)
    assert naive.x > obj.position.x + 0.02


def test_standing_things_use_the_bottom_centre_of_the_box():
    params = make_params()
    u, v = projection.ground_to_pixel(2.0, -0.4, params)
    for cls in (ROBOT, GOALPOST):
        obj = projection.project_to_ground_plane(det(cls, u - 40, v - 300, u + 40, v), params)
        assert (obj.position.x, obj.position.y) == pytest.approx((2.0, -0.4), abs=2e-3)


def test_landmark_uses_the_box_centre_on_the_floor():
    params = make_params()
    u, v = projection.ground_to_pixel(1.8, 0.3, params)
    obj = projection.project_to_ground_plane(det(LANDMARK, u - 20, v - 20, u + 20, v + 20), params)
    assert (obj.position.x, obj.position.y) == pytest.approx((1.8, 0.3), abs=2e-3)


def test_perception_object_carries_over_class_confidence_and_timestamp():
    params = make_params()
    d = det(ROBOT, CX - 30, CY, CX + 30, CY + 150, timestamp=1234.5)
    obj = projection.project_to_ground_plane(d, params)
    assert (obj.class_name, obj.confidence, obj.timestamp) == (ROBOT, 0.9, 1234.5)


def test_project_detections_skips_ones_that_cannot_reach_the_floor():
    params = make_params(pitch_deg=0.0)
    good = det(ROBOT, CX - 30, CY, CX + 30, CY + 200)
    bad = det(ROBOT, CX - 30, 0, CX + 30, 100)  # bottom edge above the horizon
    objects = projection.project_detections([bad, good], params)
    assert len(objects) == 1


def test_accuracy_target_at_two_metres_with_a_small_mount_error():
    # Milestone: error <= 0.3 m at 2 m. How wrong may the hand-measured mount be?
    true = make_params(height_m=0.50, pitch_deg=20.0)
    u, v = projection.ground_to_pixel(2.0, 0.0, true)
    believed = make_params(height_m=0.52, pitch_deg=21.0)  # 2 cm and 1 deg off
    pos = projection.pixel_to_ground(u, v, believed)
    assert math.hypot(pos.x - 2.0, pos.y) < 0.3


def test_with_mount_changes_only_the_named_fields():
    params = make_params()
    changed = projection.with_mount(params, pitch_deg=30.0)
    assert changed.mount.pitch_deg == 30.0
    assert changed.mount.height_m == params.mount.height_m
    assert params.mount.pitch_deg == 20.0  # original untouched


def write_params(path: Path, with_mount: bool) -> None:
    data = {
        "resolution": {"width": WIDTH, "height": HEIGHT},
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "distortion": [0.0] * 5,
        "reprojection_error_px": 0.2,
    }
    if with_mount:
        data["mount"] = {"height_m": 0.5, "pitch_deg": 20.0}
    path.write_text(yaml.safe_dump(data))


def test_load_camera_params_reads_calibrate_output_and_mount(tmp_path):
    write_params(tmp_path / "cam.yaml", with_mount=True)
    params = projection.load_camera_params(tmp_path / "cam.yaml")
    assert params.resolution == (WIDTH, HEIGHT)
    assert params.mount.pitch_deg == 20.0
    assert projection.pixel_to_ground(CX, CY, params).x == pytest.approx(
        projection.pixel_to_ground(CX, CY, make_params()).x
    )


def test_load_camera_params_names_what_is_missing(tmp_path):
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"distortion": [0.0] * 5}))
    with pytest.raises(ValueError, match="camera_matrix"):
        projection.load_camera_params(tmp_path / "bad.yaml")


def test_ball_radius_defaults_to_5_cm_with_a_7_cm_alternate():
    assert projection.BALL_RADIUS_M == pytest.approx(0.05)
    assert projection.BALL_RADIUS_ALT_M == pytest.approx(0.07)


def test_alternate_ball_radius_changes_the_projected_ball_position():
    params = make_params()
    u, v = projection.ground_to_pixel(1.5, 0.0, params, plane_height_m=0.07)
    ball = det(BALL, u - 15, v - 15, u + 15, v + 15)
    alt = projection.project_to_ground_plane(ball, params, projection.BALL_RADIUS_ALT_M)
    default = projection.project_to_ground_plane(ball, params)
    assert alt.position.x == pytest.approx(1.5, abs=2e-3)
    assert default.position.x > alt.position.x  # wrong (smaller) radius -> placed too far


def test_load_camera_params_needs_a_mount_from_somewhere(tmp_path):
    write_params(tmp_path / "cam.yaml", with_mount=False)
    with pytest.raises(ValueError, match="mount"):
        projection.load_camera_params(tmp_path / "cam.yaml")
    params = projection.load_camera_params(tmp_path / "cam.yaml", mount=CameraMount(0.4, 10.0))
    assert params.mount.height_m == 0.4
