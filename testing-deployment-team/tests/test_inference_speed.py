"""Tests for testing-deployment-team/benchmarks/inference_speed.py. No model
is loaded: the timing and statistics are checked with a fake predict()."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "testing-deployment-team" / "benchmarks" / "inference_speed.py"
spec = importlib.util.spec_from_file_location("inference_speed", SCRIPT)
speed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed)


def write_images(folder: Path, count: int) -> None:
    for i in range(count):
        # Each image's pixel value is its index, so tests can tell which were picked.
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), np.full((8, 8, 3), i, dtype=np.uint8))


def test_summarize_reports_avg_median_p95_and_fps():
    summary = speed.summarize([10.0] * 95 + [50.0] * 5)
    assert summary["runs"] == 100
    assert summary["avg_ms"] == pytest.approx(12.0)
    assert summary["median_ms"] == pytest.approx(10.0)
    assert summary["p95_ms"] == pytest.approx(12.0)  # linear interpolation at the 95th percentile
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 50.0
    assert summary["fps"] == pytest.approx(1000 / 12.0)


def test_summarize_rejects_no_timings():
    with pytest.raises(ValueError):
        speed.summarize([])


def test_time_predict_runs_warmup_untimed_and_cycles_frames():
    seen = []
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]
    times = speed.time_predict(lambda f: seen.append(int(f[0, 0, 0])), frames, runs=7, warmup=2)
    assert len(times) == 7
    assert all(t >= 0 for t in times)
    assert seen == [0, 1] + [0, 1, 2, 0, 1, 2, 0]


def test_time_predict_rejects_bad_input():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        speed.time_predict(lambda f: None, [frame], runs=0)
    with pytest.raises(ValueError):
        speed.time_predict(lambda f: None, [], runs=5)


def test_load_frames_spreads_evenly_and_skips_non_images(tmp_path):
    write_images(tmp_path, 10)
    (tmp_path / "notes.txt").write_text("not an image")
    frames = speed.load_frames(tmp_path, limit=5)
    assert [int(f[0, 0, 0]) for f in frames] == [0, 2, 4, 6, 8]
    assert len(speed.load_frames(tmp_path, limit=100)) == 10
    assert len(speed.load_frames(tmp_path / "frame_003.png", limit=100)) == 1


def test_load_frames_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        speed.load_frames(tmp_path, limit=5)


def test_main_writes_one_row_per_weights_file(tmp_path, monkeypatch, capsys):
    """The CLI end to end, with the Detector swapped for a fake."""

    class FakeDetector:
        def __init__(self, weights_path, conf, device):
            assert device == "cpu"

        def predict(self, image):
            return []

    monkeypatch.setitem(sys.modules, "detector", type(sys)("detector"))
    sys.modules["detector"].Detector = FakeDetector
    for name in ("a.pt", "a.onnx"):
        (tmp_path / name).write_bytes(b"")
    write_images(tmp_path, 3)
    output = tmp_path / "out" / "speed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["inference_speed.py", str(tmp_path / "a.pt"), str(tmp_path / "a.onnx")]
        + ["--images", str(tmp_path), "--runs", "4", "--warmup", "1", "--output", str(output)],
    )
    assert speed.main() == 0
    results = json.loads(output.read_text())
    assert [r["format"] for r in results["results"]] == ["pt", "onnx"]
    assert all(r["runs"] == 4 for r in results["results"])
    assert results["environment"]["device"] == "cpu"
    assert results["frame_sizes"] == ["8x8"]
    assert "a.onnx" in capsys.readouterr().out
