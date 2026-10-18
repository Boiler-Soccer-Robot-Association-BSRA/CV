"""Evaluate a trained model: precision, recall, mAP@0.5, inference speed —
and optionally a visual report of every frame.

    python detection-team/training/eval.py detection-team/models/ball_v1.pt \\
        detection-team/datasets/placeholder_ball/data.yaml --split test \\
        --report-dir detection-team/runs/eval/ball_v1_test_report

    # speed on CPU only (the Oct milestone's ">=30 FPS on a MacBook CPU")
    python detection-team/training/eval.py MODEL DATA_YAML --split test --device cpu

    # score a multi-class model on ball only (its other classes are ignored)
    python detection-team/training/eval.py detection-team/models/multiclass_v1.pt \
        detection-team/datasets/placeholder_ball/data.yaml --only-classes ball

This doesn't compute metrics itself: it turns a YOLO-format split (the
dataset contract in detection-team/README.md) into the eval harness's
manifest format and runs testing-deployment-team/tests/eval_harness.py
with our `Detector`, so detection-team and testing-deployment-team get
the same numbers from the same code. Results go to
detection-team/runs/eval/<model>_<dataset>_<split>.json, and the manifest it built
is saved next to them so the harness CLI can re-run it directly.

Precision and recall are at the detector's --conf threshold (default
0.25, the operating point we'd actually run at); mAP is computed from the
detections that pass that threshold too.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))
sys.path.insert(0, str(REPO_ROOT / "testing-deployment-team" / "tests"))

import eval_harness  # noqa: E402
from detector import DEFAULT_CONFIDENCE, Detector  # noqa: E402

RUNS_DIR = REPO_ROOT / "detection-team" / "runs" / "eval"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def label_path_for(image_path: Path) -> Path:
    """YOLO's convention: swap the last /images/ directory for /labels/ and
    the extension for .txt."""
    images, labels = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    head, sep, tail = str(image_path).rpartition(images)
    if not sep:
        raise ValueError(f"{image_path} isn't under an images/ directory")
    return Path(head + labels + tail).with_suffix(".txt")


def class_names(data: dict) -> dict[int, str]:
    names = data["names"]
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def yolo_split_to_manifest(
    data_yaml: Path, split: str, only_classes: list[str] | None = None
) -> list[dict]:
    """Read one split of a YOLO-format dataset and return it in the eval
    harness's manifest format, with absolute image paths and pixel boxes.
    With `only_classes`, ground truth of other classes is left out."""
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    if split not in data:
        raise SystemExit(f"{data_yaml} has no '{split}' split")
    image_dir = root / data[split]
    names = class_names(data)

    manifest = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_path_for(image_path)
        annotations = []
        lines = label_path.read_text().split("\n") if label_path.exists() else []
        lines = [line for line in lines if line.strip()]
        if lines:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]
            for line in lines:
                class_id, cx, cy, w, h = line.split()
                if only_classes is not None and names[int(class_id)] not in only_classes:
                    continue
                cx, cy, w, h = (
                    float(cx) * width,
                    float(cy) * height,
                    float(w) * width,
                    float(h) * height,
                )
                annotations.append(
                    {
                        "class_name": names[int(class_id)],
                        "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    }
                )
        manifest.append({"image": str(image_path.resolve()), "annotations": annotations})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path, help="Trained model weights (.pt).")
    parser.add_argument("dataset", type=Path, help="A YOLO-format data.yaml.")
    parser.add_argument("--split", default="test", help="Which split to evaluate (default test).")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default=None,
        help='e.g. "cpu", "mps"; default is ultralytics\' choice (CUDA if present, else CPU).',
    )
    parser.add_argument(
        "--only-classes",
        nargs="+",
        default=None,
        help="Score only these classes; other predictions and ground truth are ignored.",
    )
    parser.add_argument("--report-dir", type=Path, default=None, help="Also write an HTML report.")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    run_name = f"{args.weights.stem}_{args.dataset.parent.name}_{args.split}"
    if args.only_classes:
        run_name += "_only-" + "-".join(args.only_classes)
    if args.device:
        run_name += f"_{args.device}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"{run_name}_manifest.yaml"
    manifest = yolo_split_to_manifest(args.dataset, args.split, args.only_classes)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    class_map = {c: c for c in args.only_classes} if args.only_classes else None
    detector = Detector(args.weights, conf=args.conf, class_map=class_map, device=args.device)
    frame_records = [] if args.report_dir else None
    results = eval_harness.run_benchmark(manifest_path, detector, args.iou_threshold, frame_records)
    results.update(
        {
            "weights": str(args.weights),
            "dataset": str(args.dataset),
            "split": args.split,
            "conf": args.conf,
            "only_classes": args.only_classes,
            "device": str(detector.model.device),
        }
    )
    results_path = args.out_dir / f"{run_name}.json"
    results_path.write_text(json.dumps(results, indent=2))

    fps = 1000 / results["ms_per_frame"] if results["ms_per_frame"] else float("inf")
    map50 = "n/a" if results["map50"] is None else f"{results['map50']:.3f}"
    print(
        f"{args.split}: precision={results['precision']:.3f} recall={results['recall']:.3f} "
        f"mAP@{args.iou_threshold}={map50} ms/frame={results['ms_per_frame']:.1f} "
        f"({fps:.1f} FPS) frames={results['num_frames']}"
    )
    print(f"Results written to {results_path}")

    if args.report_dir:
        from eval_report import write_report

        report = write_report(
            results,
            frame_records,
            args.report_dir,
            detector_name=f"Detector (conf {args.conf})",
            weights=str(args.weights),
            manifest=f"{args.dataset} [{args.split}]",
        )
        print(f"Visual report written to {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
