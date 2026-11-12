"""Tests for testing-deployment-team/benchmarks/projection_accuracy.py.

Reference points are generated from a synthetic camera with a known pose,
so the "measured" positions are exact and any error comes from the mount
the check is told about (as it would on a real rig)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.types import BoundingBox, Detection  # noqa: E402

SCRIPT = REPO_ROOT / "testing-deployment-team" / "benchmarks" / "projection_accuracy.py"
spec = importlib.util.spec_from_file_location("projection_accuracy", SCRIPT)
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)
projection = accuracy.load_projection()

WIDTH, HEIGHT = 1920, 1200
CAMERA_MATRIX = np.array([[700.0, 0, WIDTH / 2], [0, 700.0, HEIGHT / 2], [0, 0, 1]])
TRUE_MOUNT = {"height_m": 0.5, "pitch_deg": 20.0}
FLOOR_POINTS = [(1.0, 0.0), (1.5, 0.4), (2.0, 0.0), (2.0, -0.5), (1.9, 0.3), (3.0, 0.6)]


def params_for(**mount) -> "projection.CameraParams":
    return projection.CameraParams(
        camera_matrix=CAMERA_MATRIX,
        distortion=np.zeros(5),
        mount=projection.CameraMount(**{**TRUE_MOUNT, **mount}),
        resolution=(WIDTH, HEIGHT),
    )


def pixel_rows(points=FLOOR_POINTS) -> list[dict]:
    """Rows as the CSV reader returns them, with exact pixels for the true camera."""
    rows = []
    for i, (x, y) in enumerate(points):
        u, v = projection.ground_to_pixel(x, y, params_for())
        rows.append({"name": f"p{i}", "x_m": str(x), "y_m": str(y), "u": str(u), "v": str(v)})
    return rows


def test_exact_mount_gives_near_zero_error_and_passes():
    results = accuracy.evaluate(pixel_rows(), params_for(), projection)
    assert all(r.error_m < 1e-3 for r in results)
    summary = accuracy.summarize(results)
    assert summary["target"]["verdict"] == "pass"
    assert summary["overall"]["count"] == len(FLOOR_POINTS)


def test_small_mount_error_passes_and_large_one_fails():
    rows = pixel_rows()
    small = accuracy.summarize(
        accuracy.evaluate(rows, params_for(height_m=0.52, pitch_deg=21.0), projection)
    )
    assert small["target"]["verdict"] == "pass"
    large = accuracy.summarize(accuracy.evaluate(rows, params_for(pitch_deg=26.0), projection))
    assert large["target"]["verdict"] == "fail"
    assert large["target"]["reason"].startswith("max error")


def test_no_points_near_target_range_cannot_pass():
    results = accuracy.evaluate(pixel_rows([(1.0, 0.0), (3.5, 0.0)]), params_for(), projection)
    assert accuracy.summarize(results)["target"]["verdict"] == "no_data"


def test_unprojectable_point_near_target_fails():
    rows = pixel_rows([(2.0, 0.0)])
    rows[0]["v"] = "10"  # far above the horizon
    results = accuracy.evaluate(rows, params_for(), projection)
    assert results[0].error_m is None and "not projectable" in results[0].note
    summary = accuracy.summarize(results)
    assert summary["target"]["verdict"] == "fail"
    assert summary["unprojected"][0]["name"] == "p0"


def test_errors_are_grouped_by_true_distance():
    summary = accuracy.summarize(accuracy.evaluate(pixel_rows(), params_for(), projection))
    counts = {band: stats["count"] for band, stats in summary["by_distance"].items()}
    assert counts == {"0-1 m": 0, "1-2 m": 3, "2-3 m": 2, "3+ m": 1}


def test_raised_point_uses_its_height():
    u, v = projection.ground_to_pixel(2.0, 0.0, params_for(), plane_height_m=0.05)
    row = {"name": "ball", "x_m": "2.0", "y_m": "0.0", "u": str(u), "v": str(v)}
    flat = accuracy.evaluate([row], params_for(), projection)[0]
    raised = accuracy.evaluate([row | {"height_m": "0.05"}], params_for(), projection)[0]
    assert raised.error_m < 1e-3 < flat.error_m


class FakeDetector:
    """Returns fixed detections for any image."""

    def __init__(self, detections):
        self.detections = detections

    def predict(self, image):
        return self.detections


def ball_at(x, y, confidence, half=12) -> Detection:
    u, v = projection.ground_to_pixel(x, y, params_for(), plane_height_m=projection.BALL_RADIUS_M)
    return Detection("ball", confidence, BoundingBox(u - half, v - half, u + half, v + half))


@pytest.fixture
def photo(tmp_path) -> Path:
    import cv2

    cv2.imwrite(str(tmp_path / "ball.png"), np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    return tmp_path


def test_detection_mode_projects_the_most_confident_ball(photo):
    detector = FakeDetector([ball_at(3.0, 1.0, 0.3), ball_at(2.0, 0.2, 0.9)])
    row = {"name": "b", "x_m": "2.0", "y_m": "0.2", "image": "ball.png"}
    result = accuracy.evaluate([row], params_for(), projection, detector, photo)[0]
    assert result.error_m < 2e-3
    assert "0.90" in result.note


def test_detection_mode_reports_misses_and_missing_weights(photo):
    row = {"name": "b", "x_m": "2.0", "y_m": "0.0", "image": "ball.png"}
    missed = accuracy.evaluate([row], params_for(), projection, FakeDetector([]), photo)[0]
    assert missed.error_m is None and "no ball detected" in missed.note
    no_detector = accuracy.evaluate([row], params_for(), projection)[0]
    assert "no --weights" in no_detector.note


def test_reader_rejects_rows_without_pixel_or_image(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("name,x_m,y_m,u\nbad,2.0,0.0,500\n")
    with pytest.raises(ValueError, match="line 2: needs u and v"):
        accuracy.read_reference_points(path)


def write_inputs(folder: Path, mount: dict) -> tuple[Path, Path]:
    params = folder / "camera_params.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "resolution": {"width": WIDTH, "height": HEIGHT},
                "camera_matrix": CAMERA_MATRIX.tolist(),
                "distortion": [0.0] * 5,
                "mount": mount,
            }
        )
    )
    csv_path = folder / "points.csv"
    lines = ["name,x_m,y_m,u,v"]
    lines += [f"{r['name']},{r['x_m']},{r['y_m']},{r['u']},{r['v']}" for r in pixel_rows()]
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path, params


@pytest.mark.parametrize(
    ("mount", "exit_code"), [(TRUE_MOUNT, 0), ({"height_m": 0.5, "pitch_deg": 26.0}, 1)]
)
def test_command_line_writes_json_and_sets_exit_code(tmp_path, mount, exit_code):
    csv_path, params = write_inputs(tmp_path, mount)
    output = tmp_path / "out.json"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(csv_path),
            "--params",
            str(params),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == exit_code, run.stdout + run.stderr
    assert "Milestone" in run.stdout
    report = json.loads(output.read_text())
    assert len(report["points"]) == len(FLOOR_POINTS)
    assert report["target"]["verdict"] == ("pass" if exit_code == 0 else "fail")
