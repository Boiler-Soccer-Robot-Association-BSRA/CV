"""Tests for integration-team/calibration/capture.py without a camera:
the helpers and --check on rendered board photos, and the live loop
against a fake camera that plays back rendered frames."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_calibrate import HEIGHT, WIDTH, board_image, render  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "integration-team" / "calibration" / "capture.py"
spec = importlib.util.spec_from_file_location("capture", SCRIPT)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)

# Board poses that put it in different parts of the picture (rvec, tvec in mm).
# At 1.2 m the board is small, so each pose stays inside one cell of the 3x3 map.
CENTER = ((0.15, 0.1, 0.0), (-192, -156, 1200))
TOP_LEFT = ((0.15, 0.1, 0.0), (-1000, -650, 1200))
BOTTOM_RIGHT = ((0.15, 0.1, 0.0), (576, 288, 1200))
NEAR = ((0.2, 0.1, 0.0), (-160, -110, 520))


def photo(pose) -> np.ndarray:
    return cv2.cvtColor(render(board_image(), *pose), cv2.COLOR_GRAY2BGR)


def corners_of(image: np.ndarray) -> np.ndarray:
    corners = capture.find_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), capture.PATTERN_SIZE)
    assert corners is not None
    return corners


def test_next_index_continues_after_existing_captures(tmp_path):
    assert capture.next_index(tmp_path) == 1
    for name in ("capture_001.png", "capture_007.png", "other_099.png", "capture_x.png"):
        (tmp_path / name).write_bytes(b"")
    assert capture.next_index(tmp_path) == 8


def test_corner_cells_and_missing_names():
    size = (300, 300)
    corners = np.array([[[10, 10]], [[290, 150]], [[150, 299]]], dtype=np.float32)
    cells = capture.corner_cells(corners, size)
    assert cells == {(0, 0), (1, 2), (2, 1)}
    assert capture.missing_cells(cells) == [
        "top",
        "top-right",
        "left",
        "center",
        "bottom-left",
        "bottom-right",
    ]
    all_cells = {(r, c) for r in range(3) for c in range(3)}
    assert capture.missing_cells(all_cells) == []


def test_board_fraction_and_movement():
    square = np.array([[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]], dtype=np.float32)
    assert capture.board_fraction(square, (100, 100)) == pytest.approx(0.25)
    assert capture.board_moved(square, square + 1.5) == pytest.approx(1.5)
    assert capture.board_moved(None, square) == float("inf")
    assert capture.board_moved(square, None) == float("inf")


def test_rendered_poses_land_where_the_tests_expect():
    size = (WIDTH, HEIGHT)
    assert capture.corner_cells(corners_of(photo(CENTER)), size) == {(1, 1)}
    assert capture.corner_cells(corners_of(photo(TOP_LEFT)), size) == {(0, 0)}
    assert capture.corner_cells(corners_of(photo(BOTTOM_RIGHT)), size) == {(2, 2)}
    far = capture.board_fraction(corners_of(photo(CENTER)), size)
    assert far < capture.board_fraction(corners_of(photo(NEAR)), size)


def test_preview_detection_matches_full_resolution():
    image = photo(CENTER)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preview = capture.detect_preview(gray)
    assert preview is not None
    assert capture.board_moved(preview, corners_of(image)) < 3  # same corners, roughly
    assert capture.detect_preview(np.full_like(gray, 128)) is None


def test_check_folder_reports_usable_photos_and_gaps(tmp_path):
    for i, pose in enumerate((CENTER, TOP_LEFT)):
        cv2.imwrite(str(tmp_path / f"{i}.png"), photo(pose))
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((HEIGHT, WIDTH, 3), 128, np.uint8))
    report = capture.check_folder(tmp_path)
    assert report["photos"] == 3
    assert report["usable"] == ["0.png", "1.png"]
    assert report["unusable"] == [("blank.png", "board not found")]
    assert report["sizes"] == [(WIDTH, HEIGHT)]
    assert "bottom-right" in report["missing"] and "top-left" not in report["missing"]
    problems = capture.advice(report)
    assert any("2 usable photos" in p for p in problems)
    assert any("bottom-right" in p for p in problems)


def test_advice_flags_mixed_sizes_and_missing_near_or_far():
    report = {
        "usable": ["a"] * 25,
        "sizes": [(1280, 720), (1920, 1200)],
        "missing": [],
        "board_fraction_min": 0.2,
        "board_fraction_max": 0.2,
    }
    problems = capture.advice(report)
    assert any("different sizes" in p for p in problems)
    assert any("close-up" in p for p in problems)
    assert any("far" in p for p in problems)
    good = {
        **report,
        "sizes": [(1920, 1200)],
        "board_fraction_min": 0.05,
        "board_fraction_max": 0.4,
    }
    assert capture.advice(good) == []


class FakeCamera:
    """Plays back a list of frames, one per read()."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.settings = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.settings[prop] = value

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        pass


def run_loop(monkeypatch, tmp_path, frames, keys, **overrides):
    """Run the live loop on `frames`; `keys` are pressed one per frame shown."""
    camera = FakeCamera(frames)
    pressed = list(keys)
    monkeypatch.setattr(capture.cv2, "VideoCapture", lambda device, *api: camera)
    monkeypatch.setattr(capture.cv2, "imshow", lambda name, image: None)
    monkeypatch.setattr(capture.cv2, "waitKey", lambda ms: pressed.pop(0) if pressed else ord("q"))
    monkeypatch.setattr(capture.cv2, "destroyAllWindows", lambda: None)
    args = capture.argparse.Namespace(
        folder=tmp_path / "photos",
        device=0,
        width=WIDTH,
        height=HEIGHT,
        fourcc=None,
        backend="auto",
        allow_other_size=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return capture.capture_loop(args), camera


def test_live_loop_saves_only_a_still_full_board(monkeypatch, tmp_path, capsys):
    still, blank = photo(CENTER), np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    # read() #1 is the size check; then one frame per key press.
    frames = [still, still, still, blank, photo(TOP_LEFT), photo(TOP_LEFT)]
    keys = [
        ord(" "),  # first sight of the board: not known to be still yet -> refused
        ord(" "),  # same board again: still -> saved
        ord(" "),  # no board -> refused
        ord(" "),  # board jumped to a new place -> refused
        ord(" "),  # still there -> saved
        ord("q"),
    ]
    code, camera = run_loop(monkeypatch, tmp_path, frames, keys)
    assert code == 0
    saved = sorted(p.name for p in (tmp_path / "photos").iterdir())
    assert saved == ["capture_001.png", "capture_002.png"]
    image = cv2.imread(str(tmp_path / "photos" / "capture_001.png"))
    assert image.shape == (HEIGHT, WIDTH, 3)  # full size and lossless
    assert np.array_equal(image, still)
    assert camera.settings[cv2.CAP_PROP_FRAME_WIDTH] == WIDTH
    out = capsys.readouterr().out
    assert "2 of 2 photos usable" in out and "calibrate.py" in out


def test_live_loop_refuses_the_wrong_resolution(monkeypatch, tmp_path):
    small = cv2.resize(photo(CENTER), (640, 400))
    with pytest.raises(SystemExit, match="640x400"):
        run_loop(monkeypatch, tmp_path, [small], [])
    code, _ = run_loop(monkeypatch, tmp_path, [small, small], [ord("q")], allow_other_size=True)
    assert code == 0
