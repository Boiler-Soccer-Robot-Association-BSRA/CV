"""Tests for the visual eval report (eval_report.py) and the per-frame
matching data eval_harness.py feeds it."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eval_harness  # noqa: E402
import eval_report  # noqa: E402
from dummy_detector import DummyDetector  # noqa: E402

from shared.types import BoundingBox, Detection  # noqa: E402


def ball(x_min, y_min, x_max, y_max, confidence=0.9) -> Detection:
    return Detection("ball", confidence, BoundingBox(x_min, y_min, x_max, y_max))


def gt_ball(x_min, y_min, x_max, y_max) -> dict:
    return {"class_name": "ball", "bbox": [x_min, y_min, x_max, y_max]}


def test_match_frame_indices_reports_which_ground_truth_was_matched():
    ground_truth = [gt_ball(0, 0, 10, 10), gt_ball(50, 50, 60, 60)]
    hit = ball(50, 50, 60, 60)
    stray = ball(100, 100, 110, 110, confidence=0.5)
    matches, missed = eval_harness.match_frame_indices([stray, hit], ground_truth, 0.5)
    assert matches == [(hit, 1), (stray, None)]
    assert missed == [0]


def test_match_frame_detailed_still_agrees_with_indices():
    ground_truth = [gt_ball(0, 0, 10, 10)]
    preds = [ball(0, 0, 10, 10), ball(30, 30, 40, 40, 0.3)]
    matches, fn = eval_harness.match_frame_detailed(preds, ground_truth, 0.5)
    assert [is_tp for _, is_tp in matches] == [True, False]
    assert fn == 0


def test_run_benchmark_collects_frame_records_only_when_asked():
    records = []
    results = eval_harness.run_benchmark(
        eval_harness.DEFAULT_MANIFEST, DummyDetector(), 0.5, frame_records=records
    )
    assert len(records) == results["num_frames"]
    assert {"image_path", "ground_truth", "matches", "missed", "ms"} <= set(records[0])
    assert all(Path(r["image_path"]).exists() for r in records)
    # Collecting records doesn't change the metrics (timing aside).
    plain = eval_harness.run_benchmark(eval_harness.DEFAULT_MANIFEST, DummyDetector(), 0.5)
    plain.pop("ms_per_frame")
    results.pop("ms_per_frame")
    assert plain == results


def pixel(canvas, x, y) -> tuple[int, int, int]:
    return tuple(int(v) for v in canvas[y, x])


def test_draw_evaluation_colors_boxes_by_outcome():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    ground_truth = [gt_ball(20, 120, 60, 160), gt_ball(120, 120, 160, 160)]
    correct = ball(20, 120, 60, 160)
    false_alarm = ball(120, 20, 160, 60, confidence=0.4)
    canvas = eval_report.draw_evaluation(
        image, ground_truth, [(correct, 0), (false_alarm, None)], missed=[1]
    )
    assert pixel(canvas, 20, 150) == eval_report.CORRECT_COLOR  # left edge, below the label
    assert pixel(canvas, 120, 50) == eval_report.FALSE_ALARM_COLOR
    assert pixel(canvas, 120, 150) == eval_report.MISSED_COLOR  # dashed, but starts on a dash
    assert not image.any()  # input untouched


def test_draw_evaluation_resizes_into_range_and_scales_boxes():
    small = np.zeros((64, 64, 3), dtype=np.uint8)
    canvas = eval_report.draw_evaluation(small, [], [], [], max_side=960, min_side=480)
    assert canvas.shape == (480, 480, 3)

    big = np.zeros((2000, 1000, 3), dtype=np.uint8)
    canvas = eval_report.draw_evaluation(
        big, [], [(ball(0, 1000, 500, 1990), None)], [], max_side=960, min_side=480
    )
    assert canvas.shape == (960, 480, 3)
    assert pixel(canvas, 0, 700) == eval_report.FALSE_ALARM_COLOR  # x=0, y=1400 * 0.48 ~ 672..955


def test_frame_tags():
    det = ball(0, 0, 1, 1)
    assert eval_report.frame_tags({"matches": [(det, 0)], "missed": []}) == ["correct"]
    assert eval_report.frame_tags({"matches": [], "missed": []}) == ["correct"]
    assert eval_report.frame_tags({"matches": [(det, None)], "missed": [0]}) == [
        "missed",
        "false_alarm",
    ]


def test_write_report_puts_failures_first_and_writes_every_frame(tmp_path):
    records = []
    results = eval_harness.run_benchmark(
        eval_harness.DEFAULT_MANIFEST, DummyDetector(), 0.5, frame_records=records
    )
    # Pretend the detector missed the ball in the third frame.
    records[2]["missed"] = [0]
    records[2]["matches"] = []

    index = eval_report.write_report(results, records, tmp_path, detector_name="dummy")
    page = index.read_text()

    assert len(list((tmp_path / "frames").glob("*.jpg"))) == len(records)
    assert page.count('class="card"') == len(records)
    assert page.index("frame_002.png") < page.index("frame_000.png")
    assert "Missed (1)" in page and "Correct (4)" in page


def test_main_writes_report_when_asked(tmp_path, monkeypatch):
    report_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_harness.py",
            "--output",
            str(tmp_path / "latest.json"),
            "--report-dir",
            str(report_dir),
            "--no-baseline-check",
        ],
    )
    assert eval_harness.main() == 0
    assert (report_dir / "index.html").exists()
