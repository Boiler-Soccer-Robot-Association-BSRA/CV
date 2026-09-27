import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import eval_harness
from dummy_detector import DummyDetector

from shared.types import BoundingBox, Detection


def test_iou_identical_boxes_is_one():
    box = BoundingBox(0, 0, 10, 10)
    assert eval_harness.iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(20, 20, 30, 30)
    assert eval_harness.iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 0, 15, 10)
    # intersection = 5x10 = 50, union = 100+100-50 = 150
    assert eval_harness.iou(a, b) == 50 / 150


def test_match_frame_counts_true_positive():
    pred = Detection(class_name="ball", confidence=0.9, bbox=BoundingBox(0, 0, 10, 10))
    gt = [{"class_name": "ball", "bbox": [0, 0, 10, 10]}]
    tp, fp, fn = eval_harness.match_frame([pred], gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 0, 0)


def test_match_frame_counts_false_positive_and_false_negative():
    pred = Detection(class_name="ball", confidence=0.9, bbox=BoundingBox(50, 50, 60, 60))
    gt = [{"class_name": "ball", "bbox": [0, 0, 10, 10]}]
    tp, fp, fn = eval_harness.match_frame([pred], gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_frame_ignores_class_mismatch():
    pred = Detection(class_name="robot", confidence=0.9, bbox=BoundingBox(0, 0, 10, 10))
    gt = [{"class_name": "ball", "bbox": [0, 0, 10, 10]}]
    tp, fp, fn = eval_harness.match_frame([pred], gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_frame_empty_ground_truth_and_predictions():
    tp, fp, fn = eval_harness.match_frame([], [], iou_threshold=0.5)
    assert (tp, fp, fn) == (0, 0, 0)


def test_check_against_baseline_passes_within_threshold():
    results = {"recall": 0.90}
    baseline = {"recall": 0.92}
    assert eval_harness.check_against_baseline(results, baseline, max_recall_drop=0.05)


def test_check_against_baseline_fails_beyond_threshold():
    results = {"recall": 0.80}
    baseline = {"recall": 0.92}
    assert not eval_harness.check_against_baseline(results, baseline, max_recall_drop=0.05)


def test_end_to_end_dummy_detector_on_sample_dataset():
    results = eval_harness.run_benchmark(
        eval_harness.DEFAULT_MANIFEST, DummyDetector(), iou_threshold=0.5
    )
    assert results["num_frames"] == 5
    assert results["num_ground_truth"] == 4
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["ms_per_frame"] >= 0.0


def test_main_writes_results_and_respects_baseline_gate(tmp_path):
    output_path = tmp_path / "results.json"
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"recall": 1.0}))

    argv = sys.argv
    sys.argv = [
        "eval_harness.py",
        "--output",
        str(output_path),
        "--baseline",
        str(baseline_path),
        "--max-recall-drop",
        "0.0",
    ]
    try:
        exit_code = eval_harness.main()
    finally:
        sys.argv = argv

    assert exit_code == 0
    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written["recall"] == 1.0


def test_main_fails_nonzero_when_recall_regresses(tmp_path):
    output_path = tmp_path / "results.json"
    baseline_path = tmp_path / "baseline.json"
    # An unreachable baseline forces a regression against the real dummy detector.
    baseline_path.write_text(json.dumps({"recall": 1.0}))

    class AlwaysMissDetector:
        def __init__(self, weights_path=None):
            pass

        def predict(self, image):
            return []

    detector_file = tmp_path / "always_miss_detector.py"
    detector_file.write_text(
        "class AlwaysMissDetector:\n"
        "    def __init__(self, weights_path=None):\n"
        "        pass\n"
        "    def predict(self, image):\n"
        "        return []\n"
    )

    argv = sys.argv
    sys.argv = [
        "eval_harness.py",
        "--detector-file",
        str(detector_file),
        "--detector-class",
        "AlwaysMissDetector",
        "--output",
        str(output_path),
        "--baseline",
        str(baseline_path),
        "--max-recall-drop",
        "0.0",
    ]
    try:
        exit_code = eval_harness.main()
    finally:
        sys.argv = argv

    assert exit_code == 1
