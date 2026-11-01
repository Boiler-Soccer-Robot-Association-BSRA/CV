"""Tests for integration-team/calibration/calibrate.py on rendered board photos."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "integration-team" / "calibration" / "calibrate.py"
WIDTH, HEIGHT = 960, 600
FOCAL = 500.0
SQUARE_PX = 40  # board image pixels per 35 mm square


def board_image() -> np.ndarray:
    """A 10x7-square board with a white margin."""
    margin = SQUARE_PX
    board = np.full((7 * SQUARE_PX + 2 * margin, 10 * SQUARE_PX + 2 * margin), 255, np.uint8)
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                y, x = margin + row * SQUARE_PX, margin + col * SQUARE_PX
                board[y : y + SQUARE_PX, x : x + SQUARE_PX] = 0
    return board


def render(board: np.ndarray, rvec, tvec) -> np.ndarray:
    """What a known pinhole camera sees of the board at this pose."""
    K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array(rvec, float))
    mm_per_px = 35.0 / SQUARE_PX
    board_to_camera = np.column_stack((R[:, 0] * mm_per_px, R[:, 1] * mm_per_px, tvec))
    return cv2.warpPerspective(board, K @ board_to_camera, (WIDTH, HEIGHT), borderValue=255)


def write_photos(folder: Path) -> None:
    board = board_image()
    poses = [
        ((0.3, 0.2, 0.0), (-200, -130, 500)),
        ((-0.3, 0.2, 0.1), (-200, -130, 520)),
        ((0.3, -0.3, -0.1), (-180, -120, 480)),
        ((-0.2, -0.3, 0.2), (-190, -150, 540)),
        ((0.4, 0.0, 0.1), (-210, -110, 500)),
        ((0.0, 0.4, -0.2), (-170, -130, 520)),
    ]
    folder.mkdir()
    for i, (rvec, tvec) in enumerate(poses):
        cv2.imwrite(str(folder / f"{i}.png"), render(board, rvec, tvec))
    cv2.imwrite(str(folder / "blank.png"), np.full((HEIGHT, WIDTH), 128, np.uint8))


def test_recovers_known_camera(tmp_path):
    photos, out = tmp_path / "photos", tmp_path / "camera_params.yaml"
    write_photos(photos)
    cmd = [sys.executable, str(SCRIPT), str(photos), "--output", str(out)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "skipped blank.png" in result.stdout

    params = yaml.safe_load(out.read_text())
    K = np.array(params["camera_matrix"])
    assert params["photos_used"] == 6
    assert params["resolution"] == {"width": WIDTH, "height": HEIGHT}
    assert params["reprojection_error_px"] < 0.5
    assert abs(K[0, 0] - FOCAL) / FOCAL < 0.03
    assert abs(K[1, 1] - FOCAL) / FOCAL < 0.03


def test_too_few_photos_fails(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "blank.png"), np.full((HEIGHT, WIDTH), 128, np.uint8))
    cmd = [sys.executable, str(SCRIPT), str(photos), "--output", str(tmp_path / "out.yaml")]
    assert subprocess.run(cmd, capture_output=True).returncode != 0
