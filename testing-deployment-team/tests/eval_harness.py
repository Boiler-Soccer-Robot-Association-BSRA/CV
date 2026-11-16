"""Evaluation / benchmark harness: run a detector over a labeled dataset
and report precision, recall (at IoU 0.5), and speed.

Reads a dataset manifest (YAML or JSON) of the form:

    - image: frame_000.png       # path, relative to the manifest's own directory
      annotations:
      - class_name: ball
        bbox: [x_min, y_min, x_max, y_max]
    - image: frame_001.png
      annotations: []            # a frame can have zero ground-truth objects

Runs the given detector over every frame, matches predictions to ground
truth per class by IoU >= --iou-threshold (greedy, highest-confidence
predictions matched first), and reports:

  - precision = TP / (TP + FP)
  - recall    = TP / (TP + FN)
  - map50     = mean over classes of average precision at the same IoU
                threshold (all-point interpolated, VOC/COCO style); only
                classes with at least one ground-truth box are averaged
  - per_class_ap (the per-class APs that went into map50)
  - ms_per_frame (mean wall-clock time per predict() call)

Results are written to --output as JSON. If --baseline exists and
--no-baseline-check isn't set, recall is compared against the baseline's
recall; if it has dropped by more than --max-recall-drop, this exits
nonzero so CI (or a human) catches the regression.

Three ways to run it:

  * `python eval_harness.py` (no arguments): the bundled dummy detector on
    its tiny synthetic dataset in fixtures/, gated on
    benchmarks/baseline_dummy.json. Proves the pipeline runs end to end
    (read image -> predict -> score) with no model.
  * `python eval_harness.py --recorded`: scores the real model's recorded
    detections (benchmarks/recorded/, made by record_detections.py) against
    benchmarks/baseline.json, overall and per class. This is the real-model
    regression gate CI runs: it needs no weights and no images.
  * `--detector-file / --detector-class / --weights / --manifest`: run any
    detector on any dataset you have locally; see dummy_detector.py's
    docstring for the interface. Gated on benchmarks/baseline.json unless
    --baseline or --no-baseline-check says otherwise.

Sept-Oct milestone. Coordinate with detection-team/training/eval.py so
this doesn't duplicate that work — this harness is meant to be the
shared entry point both teams run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.types import BoundingBox, Detection  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).parent / "fixtures" / "sample_dataset" / "manifest.yaml"
DEFAULT_DETECTOR_FILE = Path(__file__).parent / "dummy_detector.py"
DEFAULT_DETECTOR_CLASS = "DummyDetector"
BENCHMARKS_DIR = REPO_ROOT / "testing-deployment-team" / "benchmarks"
# The real model's scores (multiclass_v2 on the held-out test set).
DEFAULT_BASELINE = BENCHMARKS_DIR / "baseline.json"
# The bundled dummy detector's scores on the synthetic fixture frames.
DUMMY_BASELINE = BENCHMARKS_DIR / "baseline_dummy.json"
DEFAULT_RECORDED = (
    BENCHMARKS_DIR / "recorded" / "multiclass_v2_multiclass_r180_test_detections.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "testing-deployment-team" / "benchmarks" / "results" / "latest.json"


class DetectorProtocol(Protocol):
    def predict(self, image: Any) -> list[Detection]: ...


def load_manifest(path: Path) -> list[dict]:
    with path.open("r") as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def load_detector_class(file_path: Path, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load a module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x_min = max(a.x_min, b.x_min)
    y_min = max(a.y_min, b.y_min)
    x_max = min(a.x_max, b.x_max)
    y_max = min(a.y_max, b.y_max)

    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    if intersection == 0.0:
        return 0.0

    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_frame_indices(
    predictions: list[Detection], ground_truth: list[dict], iou_threshold: float
) -> tuple[list[tuple[Detection, int | None]], list[int]]:
    """Match one frame's predictions to its ground truth.

    Returns ([(prediction, matched_ground_truth_index or None), ...],
    [indices of ground-truth boxes nothing matched]). Predictions are
    matched greedily, highest confidence first, each ground-truth box at
    most once.
    """
    unmatched_gt = list(range(len(ground_truth)))
    matches = []

    for pred in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        best_iou = 0.0
        best_idx = None
        for idx in unmatched_gt:
            gt = ground_truth[idx]
            if gt["class_name"] != pred.class_name:
                continue
            gt_bbox = BoundingBox(*gt["bbox"])
            score = iou(pred.bbox, gt_bbox)
            if score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx is not None and best_iou >= iou_threshold:
            matches.append((pred, best_idx))
            unmatched_gt.remove(best_idx)
        else:
            matches.append((pred, None))

    return matches, unmatched_gt


def match_frame_detailed(
    predictions: list[Detection], ground_truth: list[dict], iou_threshold: float
) -> tuple[list[tuple[Detection, bool]], int]:
    """Like match_frame_indices, but returns
    ([(prediction, is_true_positive), ...], false_negatives)."""
    matches, unmatched_gt = match_frame_indices(predictions, ground_truth, iou_threshold)
    return [(pred, idx is not None) for pred, idx in matches], len(unmatched_gt)


def match_frame(
    predictions: list[Detection], ground_truth: list[dict], iou_threshold: float
) -> tuple[int, int, int]:
    """Return (true_positives, false_positives, false_negatives) for one frame."""
    matches, fn = match_frame_detailed(predictions, ground_truth, iou_threshold)
    tp = sum(1 for _, is_tp in matches if is_tp)
    return tp, len(matches) - tp, fn


def average_precision(scored: list[tuple[float, bool]], num_ground_truth: int) -> float:
    """All-point interpolated AP for one class.

    `scored` is (confidence, is_true_positive) for every prediction of
    that class across the whole dataset.
    """
    if num_ground_truth == 0:
        raise ValueError("AP is undefined for a class with no ground truth")

    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    precisions = []
    recalls = []
    tp = fp = 0
    for _, is_tp in scored:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / num_ground_truth)

    # Make precision monotonically non-increasing, then integrate it over
    # every point where recall changes.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def score_frames(
    frames: Iterable[dict[str, Any]],
    iou_threshold: float,
    frame_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the metrics from per-frame predictions and ground truth.

    Each frame is a dict with `predictions` (list[Detection]),
    `ground_truth` (list of {class_name, bbox}), `ms` (time the prediction
    took) and `image_path`. This is the scoring half of the harness: it
    runs no detector and reads no images, so it works the same on frames a
    detector just produced (`run_benchmark`) and on detections recorded
    earlier (`load_recorded_frames`).
    """
    total_tp = total_fp = total_fn = 0
    total_predictions = total_ground_truth = num_frames = 0
    frame_times_ms = []
    scored_by_class: dict[str, list[tuple[float, bool]]] = {}
    ground_truth_by_class: dict[str, int] = {}
    found_by_class: dict[str, int] = {}

    for frame in frames:
        predictions, ground_truth = frame["predictions"], frame["ground_truth"]
        num_frames += 1
        frame_times_ms.append(frame["ms"])
        matches, missed = match_frame_indices(predictions, ground_truth, iou_threshold)
        tp = sum(1 for _, idx in matches if idx is not None)

        total_tp += tp
        total_fp += len(matches) - tp
        total_fn += len(missed)
        for pred, idx in matches:
            is_tp = idx is not None
            scored_by_class.setdefault(pred.class_name, []).append((pred.confidence, is_tp))
        if frame_records is not None:
            frame_records.append(
                {
                    "image_path": frame["image_path"],
                    "ground_truth": ground_truth,
                    "matches": matches,
                    "missed": missed,
                    "ms": frame["ms"],
                }
            )
        for gt_index, gt in enumerate(ground_truth):
            name = gt["class_name"]
            ground_truth_by_class[name] = ground_truth_by_class.get(name, 0) + 1
            found_by_class[name] = found_by_class.get(name, 0) + (gt_index not in missed)
        total_predictions += len(predictions)
        total_ground_truth += len(ground_truth)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    ms_per_frame = sum(frame_times_ms) / len(frame_times_ms) if frame_times_ms else 0.0

    per_class_ap = {
        class_name: average_precision(scored_by_class.get(class_name, []), count)
        for class_name, count in sorted(ground_truth_by_class.items())
    }
    per_class_recall = {
        class_name: found_by_class[class_name] / count
        for class_name, count in sorted(ground_truth_by_class.items())
    }
    # With no ground truth anywhere, mAP is undefined; report null rather
    # than a number that looks meaningful.
    map50 = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else None

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "per_class_ap": per_class_ap,
        "per_class_recall": per_class_recall,
        "ms_per_frame": ms_per_frame,
        "num_frames": num_frames,
        "num_predictions": total_predictions,
        "num_ground_truth": total_ground_truth,
        "iou_threshold": iou_threshold,
    }


def run_benchmark(
    manifest_path: Path,
    detector: DetectorProtocol,
    iou_threshold: float,
    frame_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run `detector` over every frame in the manifest and compute metrics.

    If `frame_records` is a list, one dict per frame is appended to it
    (image_path, ground_truth, matches, missed, ms) — what eval_report.py
    needs to draw each frame. Left None, nothing per-frame is kept.
    """
    manifest = load_manifest(manifest_path)
    dataset_dir = manifest_path.parent

    def frames():
        for entry in manifest:
            image_path = dataset_dir / entry["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            start = time.perf_counter()
            predictions = detector.predict(image)
            yield {
                "predictions": predictions,
                "ground_truth": entry.get("annotations", []),
                "ms": (time.perf_counter() - start) * 1000,
                "image_path": image_path,
            }

    return score_frames(frames(), iou_threshold, frame_records)


# --- recorded detections ----------------------------------------------------------
# CI has no model weights and no dataset images, so it can't run a real model.
# Instead, a model's detections on the test set are recorded once on a machine
# that has both (tests/record_detections.py) and committed as two small JSON
# files, which CI scores with the same code as a live run:
#
#   labels:     {"classes": [...], "frames": {image name: [[class, x1, y1, x2, y2], ...]}}
#   detections: {"classes": [...], "frames": {image name: [[class, conf, x1, y1, x2, y2], ...]},
#                "labels": labels file name, "fingerprint": ..., "ms_per_frame": ..., ...}
#
# `class` is an index into that file's `classes`. Boxes are pixels.


def load_recorded_frames(detections_path: Path, labels_path: Path | None = None) -> list[dict]:
    """Recorded detections and their labels as frames for `score_frames`.

    `labels_path` defaults to the labels file named in the detections file,
    in the same folder. Raises ValueError if the two files don't describe
    the same split (fingerprint) or the same set of frames.
    """
    detections = json.loads(detections_path.read_text())
    if labels_path is None:
        labels_path = detections_path.parent / detections["labels"]
    labels = json.loads(labels_path.read_text())
    if detections.get("fingerprint") != labels.get("fingerprint"):
        raise ValueError(
            f"{detections_path.name} was recorded on split {detections.get('fingerprint')}, "
            f"but {labels_path.name} is split {labels.get('fingerprint')}"
        )
    if set(detections["frames"]) != set(labels["frames"]):
        raise ValueError(f"{detections_path.name} and {labels_path.name} list different frames")
    ms = float(detections.get("ms_per_frame") or 0.0)
    frames = []
    for name in sorted(labels["frames"]):
        frames.append(
            {
                "predictions": [
                    Detection(detections["classes"][int(c)], float(conf), BoundingBox(*box))
                    for c, conf, *box in detections["frames"][name]
                ],
                "ground_truth": [
                    {"class_name": labels["classes"][int(c)], "bbox": list(box)}
                    for c, *box in labels["frames"][name]
                ],
                "ms": ms,
                "image_path": Path(name),
            }
        )
    return frames


def check_against_baseline(
    results: dict[str, Any], baseline: dict[str, Any], max_recall_drop: float
) -> bool:
    """Return True if results pass the baseline gate: overall recall, and
    each class's recall when the baseline lists them (`per_class_recall`),
    must not be more than `max_recall_drop` below the baseline's."""
    return not baseline_failures(results, baseline, max_recall_drop)


def baseline_failures(
    results: dict[str, Any], baseline: dict[str, Any], max_recall_drop: float
) -> list[str]:
    """One line per recall number that dropped too far; empty if the gate passes."""
    checks = [("overall", results["recall"], baseline["recall"])]
    for name, expected in baseline.get("per_class_recall", {}).items():
        checks.append((name, results.get("per_class_recall", {}).get(name, 0.0), expected))
    return [
        f"{name} recall {actual:.3f} is more than {max_recall_drop} below baseline {expected:.3f}"
        for name, actual, expected in checks
        if expected - actual > max_recall_drop
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--detector-file", type=Path, default=DEFAULT_DETECTOR_FILE)
    parser.add_argument("--detector-class", type=str, default=DEFAULT_DETECTOR_CLASS)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Default: benchmarks/baseline.json (the real model), or "
        "benchmarks/baseline_dummy.json for the bundled dummy detector and fixture.",
    )
    parser.add_argument(
        "--recorded",
        type=Path,
        nargs="?",
        const=DEFAULT_RECORDED,
        default=None,
        metavar="DETECTIONS_JSON",
        help="Score recorded detections instead of running a detector (no weights or "
        "images needed). With no path: the committed multiclass_v2 recording.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Labels file for --recorded (default: the one the detections file names).",
    )
    parser.add_argument("--max-recall-drop", type=float, default=0.05)
    parser.add_argument(
        "--no-baseline-check",
        action="store_true",
        help="Run and report, but don't gate on the baseline.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Also write a visual HTML report (ground truth vs. predictions per frame) here.",
    )
    args = parser.parse_args()

    frame_records = [] if args.report_dir else None
    if args.recorded is not None:
        if args.report_dir:
            parser.error("--report-dir needs the images, so it can't be used with --recorded")
        frames = load_recorded_frames(args.recorded, args.labels)
        results = score_frames(frames, args.iou_threshold)
    else:
        detector_class = load_detector_class(args.detector_file, args.detector_class)
        detector = detector_class(weights_path=args.weights)
        results = run_benchmark(args.manifest, detector, args.iou_threshold, frame_records)

    if args.baseline is None:
        is_bundled_dummy = (
            args.recorded is None
            and args.detector_file == DEFAULT_DETECTOR_FILE
            and args.manifest == DEFAULT_MANIFEST
        )
        args.baseline = DUMMY_BASELINE if is_bundled_dummy else DEFAULT_BASELINE

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    map50 = "n/a" if results["map50"] is None else f"{results['map50']:.3f}"
    print(
        f"precision={results['precision']:.3f} recall={results['recall']:.3f} "
        f"mAP@{args.iou_threshold}={map50} "
        f"ms/frame={results['ms_per_frame']:.2f} frames={results['num_frames']}"
    )
    print(f"Results written to {args.output}")

    if args.report_dir:
        from eval_report import write_report

        report = write_report(
            results,
            frame_records,
            args.report_dir,
            detector_name=f"{args.detector_file.name}:{args.detector_class}",
            weights=str(args.weights) if args.weights else None,
            manifest=str(args.manifest),
        )
        print(f"Visual report written to {report}")

    if args.no_baseline_check:
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}; skipping regression check.")
        return 0

    with args.baseline.open("r") as f:
        baseline = json.load(f)

    failures = baseline_failures(results, baseline, args.max_recall_drop)
    if not failures:
        per_class = " and every class" if baseline.get("per_class_recall") else ""
        print(
            f"Recall{per_class} within {args.max_recall_drop} of baseline "
            f"{args.baseline.name} ({baseline['recall']:.3f}). OK."
        )
        return 0

    for failure in failures:
        print(f"FAIL: {failure}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
