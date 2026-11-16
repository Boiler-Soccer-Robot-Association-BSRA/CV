"""Export trained YOLO weights (.pt) to ONNX, and check the export.

Dec Week 11. ONNX runs the same model through onnxruntime instead of
PyTorch, which is faster on CPU (see the README's "Inference speed").

    # export at the image size the model was trained at (read from the .pt)
    python testing-deployment-team/deployment/export_onnx.py \\
        detection-team/models/multiclass_v2.pt --output detection-team/models/multiclass_v2.onnx

    # ...and check it against the .pt on the test set
    python testing-deployment-team/deployment/export_onnx.py \\
        detection-team/models/multiclass_v2.pt --output detection-team/models/multiclass_v2.onnx \\
        --check data-team/datasets/multiclass_r180/data.yaml --conf 0.20

Needs testing-deployment-team/requirements.txt (onnx, onnxslim, onnxruntime).

**Image size.** The export uses the size stored in the checkpoint by
training (640 for our models), not a default, so a model trained at 960
is exported at 960. `--imgsz` overrides it.

**Fixed vs. dynamic input.** By default the ONNX file takes one fixed
square input (e.g. 1x3x640x640), so every frame is padded to a square. The
.pt pads only to the next multiple of 32 (a 640x480 frame runs as
640x480), so the two see slightly different pixels and their scores differ
slightly. `--dynamic` exports a file that accepts any input size and is
run with the same padding as the .pt, which makes its detections match the
.pt's. The README's "ONNX export" section has the measured numbers for both.

**The check** (`--check DATA_YAML`) scores the .pt and the new .onnx on the
same split with the same code as detection-team/training/eval.py (the eval
harness, IoU 0.5, CPU), prints both, and exits nonzero if precision or
recall differ by more than `--max-diff` (default 0.02). Results go to
--results as JSON (gitignored).

The exported file is a model weight: never commit it (`*.onnx` is
gitignored). Share it through the team drive like the .pt.

TensorRT is not handled here. It only matters if the onboard computer is
NVIDIA-based, and its engines must be built on that machine.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

EVAL_PATH = REPO_ROOT / "detection-team" / "training" / "eval.py"
DEFAULT_RESULTS = (
    REPO_ROOT / "testing-deployment-team" / "benchmarks" / "results" / "onnx_export_check.json"
)
DEFAULT_CONF = 0.20  # the operating point in detection-team's results
DEFAULT_MAX_DIFF = 0.02
COMPARED_METRICS = ("precision", "recall", "map50")
GATED_METRICS = ("precision", "recall")


def trained_imgsz(model: Any) -> int:
    """The image size a loaded ultralytics model was trained at.

    Training stores it in the checkpoint (`train_args`), and ultralytics
    copies it into `model.overrides` on load. Raises ValueError if neither
    has it, rather than silently exporting at a default size.
    """
    candidates = [
        (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz"),
        (getattr(model, "overrides", None) or {}).get("imgsz"),
    ]
    for value in candidates:
        if isinstance(value, list | tuple):  # [h, w]; ours are square
            value = max(value)
        if value:
            return int(value)
    raise ValueError("The checkpoint doesn't record its training image size; pass --imgsz")


def export(
    weights: Path, output: Path, imgsz: int | None = None, dynamic: bool = False
) -> dict[str, Any]:
    """Export `weights` to ONNX at `output`. Returns what was exported:
    path, imgsz, dynamic, input shape, opset and size.

    Ultralytics always writes the .onnx next to the .pt, so the export runs
    on a temporary copy: an existing .onnx beside the weights is never
    overwritten unless it is `output` itself.
    """
    import onnx
    from ultralytics import YOLO

    if weights.suffix != ".pt":
        raise ValueError(f"Expected .pt weights, got {weights}")
    if output.suffix != ".onnx":
        raise ValueError(f"--output must end in .onnx, got {output}")

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / weights.name
        shutil.copy2(weights, copy)
        model = YOLO(str(copy))
        if imgsz is None:
            imgsz = trained_imgsz(model)
        exported = Path(model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, device="cpu"))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), output)

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    return {
        "weights": str(weights),
        "output": str(output),
        "imgsz": imgsz,
        "dynamic": dynamic,
        # A fixed dimension is a number; a dynamic one is a name like "height".
        "input_shape": [d.dim_value or d.dim_param for d in dims],
        "opset": onnx_model.opset_import[0].version,
        "size_mb": round(output.stat().st_size / 1e6, 2),
    }


def compare(pt: dict[str, Any], onnx_: dict[str, Any], max_diff: float) -> dict[str, Any]:
    """Differences (onnx - pt) for each compared metric, and whether
    precision and recall are both within `max_diff`."""
    diffs = {m: onnx_[m] - pt[m] for m in COMPARED_METRICS if pt.get(m) is not None}
    return {
        "diff": diffs,
        "max_diff": max_diff,
        "passed": all(abs(diffs[m]) <= max_diff for m in GATED_METRICS),
    }


def evaluate(weights: Path, manifest_path: Path, conf: float, iou_threshold: float) -> dict:
    """Score one weights file on a manifest with the eval harness, on CPU."""
    import eval_harness
    from detector import Detector

    detector = Detector(weights, conf=conf, device="cpu")
    results = eval_harness.run_benchmark(manifest_path, detector, iou_threshold)
    results["weights"] = str(weights)
    return results


def check_export(
    weights: Path,
    output: Path,
    data_yaml: Path,
    split: str,
    conf: float,
    iou_threshold: float,
    max_diff: float,
    work_dir: Path,
) -> dict[str, Any]:
    """Score the .pt and the .onnx on one split and compare them."""
    import yaml

    spec = importlib.util.spec_from_file_location("detection_eval", EVAL_PATH)
    detection_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detection_eval)

    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / f"{weights.stem}_{data_yaml.parent.name}_{split}_manifest.yaml"
    manifest = detection_eval.yolo_split_to_manifest(data_yaml, split)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    pt = evaluate(weights, manifest_path, conf, iou_threshold)
    onnx_ = evaluate(output, manifest_path, conf, iou_threshold)
    return {
        "dataset": str(data_yaml),
        "split": split,
        "conf": conf,
        "iou_threshold": iou_threshold,
        "device": "cpu",
        "pt": pt,
        "onnx": onnx_,
        **compare(pt, onnx_, max_diff),
    }


def format_check(check: dict[str, Any]) -> str:
    def row(name: str, r: dict[str, Any]) -> str:
        fps = 1000 / r["ms_per_frame"] if r["ms_per_frame"] else float("inf")
        return (
            f"{name:<6} {r['precision']:>9.3f} {r['recall']:>7.3f} {r['map50']:>7.3f} "
            f"{r['num_predictions']:>12} {r['ms_per_frame']:>9.1f} {fps:>6.1f}"
        )

    diff = check["diff"]
    lines = [
        f"{check['dataset']} [{check['split']}], conf {check['conf']}, "
        f"IoU {check['iou_threshold']}, {check['device']}, {check['pt']['num_frames']} frames",
        f"{'':<6} {'precision':>9} {'recall':>7} {'mAP50':>7} {'predictions':>12} "
        f"{'ms/frame':>9} {'FPS':>6}",
        row(".pt", check["pt"]),
        row(".onnx", check["onnx"]),
        f"{'diff':<6} {diff['precision']:>+9.3f} {diff['recall']:>+7.3f} {diff['map50']:>+7.3f}",
    ]
    verdict = "PASS" if check["passed"] else "FAIL"
    lines.append(
        f"{verdict}: precision and recall "
        f"{'are' if check['passed'] else 'are not both'} within {check['max_diff']} of the .pt"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path, help="Trained weights (.pt).")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the .onnx.")
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Default: the size the model was trained at."
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Accept any input size (same padding as the .pt) instead of a fixed square.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        metavar="DATA_YAML",
        help="Also score the .pt and the .onnx on this YOLO dataset and compare them.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-diff", type=float, default=DEFAULT_MAX_DIFF)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not args.weights.exists():
        parser.error(f"weights not found: {args.weights}")
    if args.check is not None and not args.check.exists():
        parser.error(f"dataset not found: {args.check}")

    info = export(args.weights, args.output, args.imgsz, args.dynamic)
    print(
        f"Exported {info['weights']} -> {info['output']} "
        f"(imgsz {info['imgsz']}, input {info['input_shape']}, opset {info['opset']}, "
        f"{info['size_mb']} MB)"
    )
    results: dict[str, Any] = {"export": info}
    passed = True
    if args.check is not None:
        check = check_export(
            args.weights,
            args.output,
            args.check,
            args.split,
            args.conf,
            args.iou_threshold,
            args.max_diff,
            args.results.parent,
        )
        print(format_check(check))
        results["check"] = check
        passed = check["passed"]

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results written to {args.results}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
