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

Defaults to the bundled dummy detector and its tiny synthetic dataset in
fixtures/, so `python eval_harness.py` with no arguments runs the whole
pipeline end to end today. Point --detector-file / --detector-class /
--weights / --manifest at a real model and dataset once one exists; see
dummy_detector.py's docstring for the exact interface required.

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
DEFAULT_BASELINE = REPO_ROOT / "testing-deployment-team" / "benchmarks" / "baseline.json"
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


def match_frame_detailed(
    predictions: list[Detection], ground_truth: list[dict], iou_threshold: float
) -> tuple[list[tuple[Detection, bool]], int]:
    """Match one frame's predictions to its ground truth.

    Returns ([(prediction, is_true_positive), ...], false_negatives).
    Predictions are matched greedily, highest confidence first, each
    ground-truth box at most once.
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
            matches.append((pred, True))
            unmatched_gt.remove(best_idx)
        else:
            matches.append((pred, False))

    return matches, len(unmatched_gt)


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


def run_benchmark(
    manifest_path: Path, detector: DetectorProtocol, iou_threshold: float
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset_dir = manifest_path.parent

    total_tp = total_fp = total_fn = 0
    total_predictions = total_ground_truth = 0
    frame_times_ms = []
    scored_by_class: dict[str, list[tuple[float, bool]]] = {}
    ground_truth_by_class: dict[str, int] = {}

    for entry in manifest:
        image_path = dataset_dir / entry["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        start = time.perf_counter()
        predictions = detector.predict(image)
        frame_times_ms.append((time.perf_counter() - start) * 1000)

        ground_truth = entry.get("annotations", [])
        matches, fn = match_frame_detailed(predictions, ground_truth, iou_threshold)
        tp = sum(1 for _, is_tp in matches if is_tp)

        total_tp += tp
        total_fp += len(matches) - tp
        total_fn += fn
        for pred, is_tp in matches:
            scored_by_class.setdefault(pred.class_name, []).append((pred.confidence, is_tp))
        for gt in ground_truth:
            ground_truth_by_class[gt["class_name"]] = (
                ground_truth_by_class.get(gt["class_name"], 0) + 1
            )
        total_predictions += len(predictions)
        total_ground_truth += len(ground_truth)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    ms_per_frame = sum(frame_times_ms) / len(frame_times_ms) if frame_times_ms else 0.0

    per_class_ap = {
        class_name: average_precision(scored_by_class.get(class_name, []), count)
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
        "ms_per_frame": ms_per_frame,
        "num_frames": len(manifest),
        "num_predictions": total_predictions,
        "num_ground_truth": total_ground_truth,
        "iou_threshold": iou_threshold,
    }


def check_against_baseline(
    results: dict[str, Any], baseline: dict[str, Any], max_recall_drop: float
) -> bool:
    """Return True if results pass the baseline gate."""
    drop = baseline["recall"] - results["recall"]
    return drop <= max_recall_drop


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
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--max-recall-drop", type=float, default=0.05)
    parser.add_argument(
        "--no-baseline-check",
        action="store_true",
        help="Run and report, but don't gate on the baseline.",
    )
    args = parser.parse_args()

    detector_class = load_detector_class(args.detector_file, args.detector_class)
    detector = detector_class(weights_path=args.weights)

    results = run_benchmark(args.manifest, detector, args.iou_threshold)

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

    if args.no_baseline_check:
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}; skipping regression check.")
        return 0

    with args.baseline.open("r") as f:
        baseline = json.load(f)

    if check_against_baseline(results, baseline, args.max_recall_drop):
        print(f"Recall within {args.max_recall_drop} of baseline ({baseline['recall']:.3f}). OK.")
        return 0

    print(
        f"FAIL: recall {results['recall']:.3f} is more than {args.max_recall_drop} below "
        f"baseline {baseline['recall']:.3f}."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
