"""The floor-grid CSV templates must stay usable by projection_accuracy.py.

The pixel template ships with empty u/v columns (they are read off a photo
on the day). Here they are filled in from a synthetic camera, standing in
for that photo, and the whole file is run through the accuracy check."""

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "testing-deployment-team" / "benchmarks"
GRID_DIR = BENCHMARKS / "floor_grid"
PIXEL_CSV = GRID_DIR / "reference_points_pixel.csv"
BALL_CSV = GRID_DIR / "reference_points_ball.csv"

spec = importlib.util.spec_from_file_location(
    "projection_accuracy", BENCHMARKS / "projection_accuracy.py"
)
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)
projection = accuracy.load_projection()

# Roughly the See3CAM_24CUG with e-con's lens: 1920x1200, ~105 degrees across.
WIDTH, HEIGHT = 1920, 1200
FOCAL = (WIDTH / 2) / math.tan(math.radians(104.6 / 2))
PARAMS = projection.CameraParams(
    camera_matrix=np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]]),
    distortion=np.zeros(5),
    mount=projection.CameraMount(height_m=0.5, pitch_deg=20.0),
    resolution=(WIDTH, HEIGHT),
)


def template_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_pixel_template_has_positions_filled_and_pixels_left_blank():
    rows = template_rows(PIXEL_CSV)
    assert list(rows[0]) == ["name", "x_m", "y_m", "u", "v"]
    assert len(rows) == 13
    assert len({r["name"] for r in rows}) == 13
    assert all(r["u"] == "" and r["v"] == "" for r in rows)
    xs = [float(r["x_m"]) for r in rows]
    ys = [float(r["y_m"]) for r in rows]
    assert min(xs) == 0.5 and max(xs) == 2.5
    assert min(ys) < 0 < max(ys)  # points on both sides
    # Left is +y: names starting with L are on the left, R on the right.
    for row in rows:
        side = float(row["y_m"])
        assert (side > 0) == row["name"].startswith("L") and (side < 0) == row["name"].startswith(
            "R"
        )


def test_unfilled_pixel_template_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="needs u and v"):
        accuracy.read_reference_points(PIXEL_CSV)


def test_filled_pixel_template_runs_and_can_decide_the_milestone(tmp_path):
    rows = template_rows(PIXEL_CSV)
    for row in rows:
        u, v = projection.ground_to_pixel(float(row["x_m"]), float(row["y_m"]), PARAMS)
        # Every point of the layout should be inside this camera's picture.
        assert 0 <= u < WIDTH and 0 <= v < HEIGHT, row["name"]
        row["u"], row["v"] = f"{u:.1f}", f"{v:.1f}"
    filled = tmp_path / "filled.csv"
    with filled.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    results = accuracy.evaluate(accuracy.read_reference_points(filled), PARAMS, projection)
    summary = accuracy.summarize(results)
    assert summary["overall"]["count"] == 13
    assert summary["overall"]["max_m"] < 0.01  # only the 0.1 px rounding above
    assert summary["target"]["verdict"] == "pass"
    assert "over 5 points near 2 m" in summary["target"]["reason"]


def test_ball_template_is_complete_and_matches_the_pixel_layout():
    ball_rows = accuracy.read_reference_points(BALL_CSV)  # valid as shipped
    grid = {r["name"]: (r["x_m"], r["y_m"]) for r in template_rows(PIXEL_CSV)}
    assert len(ball_rows) == 7
    for row in ball_rows:
        point = row["name"].removeprefix("ball_")
        assert grid[point] == (row["x_m"], row["y_m"])
        assert row["image"] == f"photos/ball_{point}.jpg"
    near_two = [
        r for r in ball_rows if abs(math.hypot(float(r["x_m"]), float(r["y_m"])) - 2) <= 0.25
    ]
    assert len(near_two) >= 3
