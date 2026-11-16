"""Tests for the real-model regression gate: recorded detections
(record_detections.py) replayed through eval_harness.py --recorded.

Two kinds of test: the mechanism on tiny made-up recordings, and the
committed multiclass_v2 recording against the committed baseline, which is
exactly what CI's gate step runs."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import eval_harness  # noqa: E402
import record_detections  # noqa: E402

from shared.types import BoundingBox, Detection  # noqa: E402

CLASSES = ["ball", "robot"]
BALL, ROBOT = [10.0, 10.0, 30.0, 30.0], [40.0, 5.0, 60.0, 45.0]


class FakeDetector:
    """Finds the robot in every frame and the ball in all but the first."""

    def __init__(self) -> None:
        self.frame = 0

    def predict(self, image) -> list[Detection]:
        self.frame += 1
        found = [Detection("robot", 0.91234567, BoundingBox(*ROBOT))]
        if self.frame > 1:
            found.append(Detection("ball", 0.8, BoundingBox(10.004, 10.0, 30.0, 30.0)))
        return found


def make_recording(tmp_path: Path, frames: int = 4, fingerprint: str = "abc123"):
    """Record FakeDetector on a tiny dataset; returns (detections path, labels path)."""
    manifest = []
    for i in range(frames):
        path = tmp_path / f"frame_{i}.png"
        cv2.imwrite(str(path), np.zeros((50, 80, 3), dtype=np.uint8))
        annotations = [{"class_name": "ball", "bbox": BALL}, {"class_name": "robot", "bbox": ROBOT}]
        manifest.append({"image": str(path), "annotations": annotations})
    labels, detections, ms = record_detections.record(manifest, FakeDetector(), CLASSES)
    common = {"dataset": "tiny", "split": "test", "fingerprint": fingerprint}
    labels_path = tmp_path / "tiny_test_labels.json"
    detections_path = tmp_path / "fake_tiny_test_detections.json"
    labels_path.write_text(record_detections.dump({**common, "classes": CLASSES}, labels))
    header = {**common, "labels": labels_path.name, "ms_per_frame": ms, "classes": CLASSES}
    detections_path.write_text(record_detections.dump(header, detections))
    return detections_path, labels_path


def test_recording_round_trips_and_scores_like_a_live_run(tmp_path):
    detections_path, labels_path = make_recording(tmp_path)
    recorded = json.loads(detections_path.read_text())
    assert recorded["frames"]["frame_0.png"] == [[1, 0.9123, 40.0, 5.0, 60.0, 45.0]]
    assert recorded["frames"]["frame_1.png"][1] == [0, 0.8, 10.0, 10.0, 30.0, 30.0]  # rounded

    frames = eval_harness.load_recorded_frames(detections_path)  # finds the labels itself
    results = eval_harness.score_frames(frames, iou_threshold=0.5)
    assert results["num_frames"] == 4
    assert results["precision"] == 1.0
    assert results["recall"] == pytest.approx(7 / 8)
    assert results["per_class_recall"] == {"ball": pytest.approx(3 / 4), "robot": 1.0}

    # The same numbers a live run of the same detector gives.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "image": f"frame_{i}.png",
                    "annotations": [
                        {"class_name": "ball", "bbox": BALL},
                        {"class_name": "robot", "bbox": ROBOT},
                    ],
                }
                for i in range(4)
            ]
        )
    )
    live = eval_harness.run_benchmark(manifest_path, FakeDetector(), iou_threshold=0.5)
    for key in ("precision", "recall", "map50", "per_class_ap", "per_class_recall"):
        assert live[key] == results[key], key


def test_mismatched_recordings_are_refused(tmp_path):
    detections_path, labels_path = make_recording(tmp_path)
    labels = json.loads(labels_path.read_text())

    other_split = tmp_path / "other_split.json"
    other_split.write_text(json.dumps({**labels, "fingerprint": "different"}))
    with pytest.raises(ValueError, match="split"):
        eval_harness.load_recorded_frames(detections_path, other_split)

    labels["frames"].pop("frame_0.png")
    fewer_frames = tmp_path / "fewer_frames.json"
    fewer_frames.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="different frames"):
        eval_harness.load_recorded_frames(detections_path, fewer_frames)


def test_gate_checks_every_class_the_baseline_lists():
    baseline = {"recall": 0.85, "per_class_recall": {"ball": 0.79, "robot": 0.88}}
    fine = {"recall": 0.84, "per_class_recall": {"ball": 0.76, "robot": 0.90}}
    assert eval_harness.check_against_baseline(fine, baseline, 0.05)

    # Overall recall holds up, but one class has dropped: the gate must fail.
    ball_dropped = {"recall": 0.84, "per_class_recall": {"ball": 0.70, "robot": 0.95}}
    failures = eval_harness.baseline_failures(ball_dropped, baseline, 0.05)
    assert len(failures) == 1 and failures[0].startswith("ball recall 0.700")

    # A class the model no longer reports at all counts as recall 0.
    missing = {"recall": 0.84, "per_class_recall": {"robot": 0.95}}
    assert not eval_harness.check_against_baseline(missing, baseline, 0.05)

    # A baseline without per-class numbers (the dummy one) gates on overall recall only.
    assert eval_harness.check_against_baseline(ball_dropped, {"recall": 0.85}, 0.05)


def run_main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["eval_harness.py", *argv])
    return eval_harness.main()


def test_main_recorded_mode_passes_and_fails_on_the_baseline(tmp_path, monkeypatch, capsys):
    detections_path, _ = make_recording(tmp_path)
    output = tmp_path / "out.json"
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"recall": 0.875, "per_class_recall": {"ball": 0.75}}))
    args = ["--recorded", str(detections_path), "--output", str(output)]
    assert run_main(monkeypatch, *args, "--baseline", str(good)) == 0
    assert json.loads(output.read_text())["per_class_recall"]["ball"] == 0.75
    assert "every class" in capsys.readouterr().out

    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps({"recall": 0.875, "per_class_recall": {"ball": 0.95}}))
    assert run_main(monkeypatch, *args, "--baseline", str(strict)) == 1
    assert "FAIL: ball recall 0.750" in capsys.readouterr().out


def test_default_baselines_are_the_dummy_one_for_the_dummy_and_the_real_one_otherwise(
    tmp_path, monkeypatch, capsys
):
    output = ["--output", str(tmp_path / "out.json")]
    assert run_main(monkeypatch, *output) == 0
    assert "baseline_dummy.json" in capsys.readouterr().out
    assert run_main(monkeypatch, "--recorded", *output) == 0
    assert "baseline baseline.json" in capsys.readouterr().out


# --- the committed recording: what CI's gate step runs ----------------------------


def test_committed_recording_matches_the_committed_baseline():
    baseline = json.loads(eval_harness.DEFAULT_BASELINE.read_text())
    frames = eval_harness.load_recorded_frames(eval_harness.DEFAULT_RECORDED)
    results = eval_harness.score_frames(frames, baseline["iou_threshold"])
    for key in ("precision", "recall", "map50"):
        assert results[key] == pytest.approx(baseline[key], abs=1e-9), key
    assert results["per_class_recall"] == pytest.approx(baseline["per_class_recall"], abs=1e-9)
    assert results["num_frames"] == baseline["num_frames"] == 1405
    assert results["num_ground_truth"] == baseline["num_ground_truth"]
    # The real model, not the dummy.
    assert set(baseline["per_class_recall"]) == {"ball", "goalpost", "landmark", "robot"}
    assert "multiclass_v2" in baseline["detector"]


def test_gate_would_catch_a_model_that_lost_a_class():
    """Drop every recorded robot detection: the per-class gate must fail on robot."""
    baseline = json.loads(eval_harness.DEFAULT_BASELINE.read_text())
    frames = eval_harness.load_recorded_frames(eval_harness.DEFAULT_RECORDED)
    for frame in frames:
        frame["predictions"] = [d for d in frame["predictions"] if d.class_name != "robot"]
    results = eval_harness.score_frames(frames, 0.5)
    failures = eval_harness.baseline_failures(results, baseline, 0.05)
    assert any(f.startswith("robot recall 0.000") for f in failures)


def test_recorded_files_stay_small_enough_to_commit():
    for path in eval_harness.DEFAULT_RECORDED.parent.glob("*.json"):
        assert path.stat().st_size < 500_000, f"{path.name} is over the 500 kB pre-commit limit"
