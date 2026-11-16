"""Record a model's detections on a test split, for the CI regression gate.

CI can't run our real model: weights and TORSO-21 images aren't in git.
So the model is run here, on a machine that has both, and what it detected
is committed as two small JSON files. CI then scores those files with the
eval harness (`eval_harness.py --recorded`) and gates on
benchmarks/baseline.json. No images or weights are written, only boxes.

    python testing-deployment-team/tests/record_detections.py \\
        detection-team/models/multiclass_v2.pt \\
        data-team/datasets/multiclass_r180/data.yaml --conf 0.20

writes, in testing-deployment-team/benchmarks/recorded/:

    multiclass_r180_test_labels.json                    the split's ground truth
    multiclass_v2_multiclass_r180_test_detections.json  what the model found

**When to re-run it**

* *A new model is going on the robot:* record it, then run
  `eval_harness.py --recorded <its detections file>`. If it passes the
  gate, point CI at the new file (DEFAULT_RECORDED in eval_harness.py) and,
  with `--write-baseline`, make its scores the new baseline. Commit the
  baseline change on its own so it's easy to see when and why it moved.
* *The test split or its labels changed* (new fingerprint in
  split_info.yaml): record again with `--write-baseline`; old recordings
  no longer match the labels, and the harness refuses to mix them.
* *The `Detector` or its dependencies changed* (e.g. an ultralytics
  upgrade): record again without `--write-baseline` and check the gate
  still passes. CI can't catch this by itself, since it runs no inference.

Boxes are rounded to 0.01 px and confidences to 4 decimals, which doesn't
change any score.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

import eval_harness  # noqa: E402

from shared.types import Detection  # noqa: E402

RECORDED_DIR = REPO_ROOT / "testing-deployment-team" / "benchmarks" / "recorded"
EVAL_PATH = REPO_ROOT / "detection-team" / "training" / "eval.py"
DEFAULT_CONF = 0.20  # the operating point in detection-team's results


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_rows(annotations: list[dict], classes: list[str]) -> list[list[float]]:
    return [
        [classes.index(a["class_name"]), *(round(float(v), 2) for v in a["bbox"])]
        for a in annotations
    ]


def detection_rows(detections: list[Detection], classes: list[str]) -> list[list[float]]:
    rows = []
    for d in detections:
        box = d.bbox
        rows.append(
            [
                classes.index(d.class_name),
                round(float(d.confidence), 4),
                *(round(float(v), 2) for v in (box.x_min, box.y_min, box.x_max, box.y_max)),
            ]
        )
    return rows


def record(manifest: list[dict], detector: Any, classes: list[str]) -> tuple[dict, dict, float]:
    """Run `detector` over the manifest. Returns ({image name: label rows},
    {image name: detection rows}, mean ms per frame)."""
    labels: dict[str, list] = {}
    detections: dict[str, list] = {}
    total_ms = 0.0
    for entry in manifest:
        image_path = Path(entry["image"])
        if image_path.name in labels:
            raise ValueError(f"Two frames are both named {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        start = time.perf_counter()
        predictions = detector.predict(image)
        total_ms += (time.perf_counter() - start) * 1000
        labels[image_path.name] = label_rows(entry.get("annotations", []), classes)
        detections[image_path.name] = detection_rows(predictions, classes)
    return labels, detections, total_ms / len(manifest) if manifest else 0.0


def dump(header: dict[str, Any], frames: dict[str, list]) -> str:
    """JSON with one frame per line: small, and a changed frame is one line
    in a git diff."""
    lines = [json.dumps(header, indent=2)[:-2] + ","]
    lines.append('  "frames": {')
    names = sorted(frames)
    for i, name in enumerate(names):
        comma = "," if i < len(names) - 1 else ""
        rows = json.dumps(frames[name], separators=(",", ":"))
        lines.append(f"    {json.dumps(name)}: {rows}{comma}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def baseline_from(results: dict[str, Any], detections_header: dict[str, Any]) -> dict[str, Any]:
    """What goes in benchmarks/baseline.json for a recording."""
    return {
        **{k: results[k] for k in ("precision", "recall", "map50")},
        "per_class_recall": results["per_class_recall"],
        "per_class_ap": results["per_class_ap"],
        "ms_per_frame": round(results["ms_per_frame"], 1),
        "num_frames": results["num_frames"],
        "num_predictions": results["num_predictions"],
        "num_ground_truth": results["num_ground_truth"],
        "iou_threshold": results["iou_threshold"],
        "detector": f"{detections_header['model']} (conf {detections_header['conf']})",
        "dataset": f"{detections_header['dataset']} [{detections_header['split']}], "
        f"split fingerprint {detections_header['fingerprint']}",
        "recorded": detections_header["recorded"],
        "note": "Scores of the real model on the held-out test set, as scored (the ball "
        "count-once rule for recording 151 is not applied here). Computed from the recorded "
        "detections in benchmarks/recorded/ by tests/record_detections.py --write-baseline. "
        "ms_per_frame is from the recording machine and is not gated.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path, help="Model weights (.pt or .onnx).")
    parser.add_argument("dataset", type=Path, help="A YOLO-format data.yaml.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=RECORDED_DIR)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Also make this recording's scores the baseline (benchmarks/baseline.json).",
    )
    parser.add_argument("--baseline", type=Path, default=eval_harness.DEFAULT_BASELINE)
    args = parser.parse_args()

    from detector import Detector

    spec = importlib.util.spec_from_file_location("detection_eval", EVAL_PATH)
    detection_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detection_eval)

    data = yaml.safe_load(args.dataset.read_text())
    classes = [name for _, name in sorted(detection_eval.class_names(data).items())]
    split_info_path = args.dataset.parent / "split_info.yaml"
    split_info = yaml.safe_load(split_info_path.read_text()) if split_info_path.exists() else {}
    fingerprint = split_info.get("fingerprint")
    if fingerprint is None:
        print(f"Warning: no fingerprint in {split_info_path}; the split can't be identified.")

    manifest = detection_eval.yolo_split_to_manifest(args.dataset, args.split)
    detector = Detector(args.weights, conf=args.conf, device=args.device)
    labels, detections, ms_per_frame = record(manifest, detector, classes)

    dataset_name = args.dataset.parent.name
    labels_name = f"{dataset_name}_{args.split}_labels.json"
    detections_name = f"{args.weights.stem}_{dataset_name}_{args.split}_detections.json"
    common = {"dataset": dataset_name, "split": args.split, "fingerprint": fingerprint}
    labels_header = {**common, "label_fixes_applied": split_info.get("label_fixes_applied")}
    detections_header = {
        "model": args.weights.stem,
        "weights_file": args.weights.name,
        "weights_sha256": sha256(args.weights),
        **common,
        "labels": labels_name,
        "conf": args.conf,
        "device": args.device,
        "ms_per_frame": round(ms_per_frame, 1),
        "recorded": datetime.date.today().isoformat(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out_dir / labels_name
    detections_path = args.out_dir / detections_name
    labels_path.write_text(dump({**labels_header, "classes": classes}, labels))
    detections_path.write_text(dump({**detections_header, "classes": classes}, detections))

    # Score what was just written, through the same path CI uses.
    frames = eval_harness.load_recorded_frames(detections_path, labels_path)
    results = eval_harness.score_frames(frames, args.iou_threshold)
    print(
        f"{len(frames)} frames: precision={results['precision']:.3f} "
        f"recall={results['recall']:.3f} mAP@{args.iou_threshold}={results['map50']:.3f}"
    )
    print(
        "recall per class: "
        + ", ".join(f"{k} {v:.3f}" for k, v in results["per_class_recall"].items())
    )
    for path in (labels_path, detections_path):
        print(f"Wrote {path} ({path.stat().st_size / 1000:.0f} kB)")
    if args.write_baseline:
        baseline = baseline_from(results, detections_header)
        args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Wrote baseline {args.baseline}")
    else:
        print("Baseline not changed (pass --write-baseline to set it from this recording).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
