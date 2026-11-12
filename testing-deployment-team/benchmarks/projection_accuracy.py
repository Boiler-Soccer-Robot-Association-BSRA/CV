"""Projection accuracy check: how far are projected positions from measured ones?

testing-deployment-team, Weeks 8-9. Compares integration-team's
ground-plane projection (integration-team/geometry/projection.py) against
reference points whose floor position was measured with a tape measure,
and checks the milestone in docs/milestones.md: position error <= 0.3 m
at 2 m range.

    python testing-deployment-team/benchmarks/projection_accuracy.py \\
        reference_points.csv --params camera_params.yaml

The reference CSV has one row per measured point, in one of two modes:

* **Pixel mode** (checks calibration + mount + projection math): you
  already know where the point is in the image.

      name,x_m,y_m,u,v
      tape_1m,1.00,0.00,962,811
      tape_2m_left,2.00,0.50,701,642

* **Detection mode** (checks the full detect -> project prototype): a
  photo with a ball placed at the measured spot. The detector finds the
  ball and its box is projected exactly as the robot would.

      name,x_m,y_m,image
      ball_1m,1.00,0.00,photos/ball_1m.jpg

  Needs --weights (e.g. ball_v4.pt from the team drive). If a photo has
  several detections of the class, the most confident one is used.

Columns:

| Column        | Meaning                                                          |
|---------------|------------------------------------------------------------------|
| `name`        | Optional label for the report                                    |
| `x_m`, `y_m`  | Measured floor position, robot frame (+x forward, +y left), m    |
| `u`, `v`      | Pixel of the point (pixel mode)                                  |
| `image`       | Photo path, relative to the CSV's folder (detection mode)        |
| `height_m`    | Optional, pixel mode: height of the marked point above the floor |
|               | (0 for tape on the floor; the ball radius for a ball's center)   |
| `class_name`  | Optional, detection mode: which class to use (default `ball`)    |

`x_m`/`y_m` are measured from the robot's origin, the point on the floor
the camera mount's `x_m`/`y_m` are measured from (directly under the lens
if both are 0). For a ball, measure to where it touches the floor.

To get pixels for pixel mode: take one photo with the camera at its real
height and tilt, then read each tape mark's pixel in any image viewer that
shows cursor coordinates (MS Paint shows them in its status bar).

Reports every point's error, error by distance band, and the milestone
check. Writes JSON to --output and exits nonzero if the check fails or
can't be made, so it can gate a run like eval_harness.py does.
"""

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.classes import BALL  # noqa: E402

PROJECTION_PATH = REPO_ROOT / "integration-team" / "geometry" / "projection.py"
DEFAULT_OUTPUT = (
    REPO_ROOT / "testing-deployment-team" / "benchmarks" / "results" / "projection_accuracy.json"
)

TARGET_ERROR_M = 0.3
TARGET_RANGE_M = 2.0
TARGET_BAND_M = 0.25
DISTANCE_BANDS_M = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, math.inf)]


def load_projection():
    """integration-team's projection module (its folder name has a hyphen,
    so it can't be imported as a package)."""
    spec = importlib.util.spec_from_file_location("projection", PROJECTION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class PointResult:
    name: str
    x_true: float
    y_true: float
    x_pred: float | None
    y_pred: float | None
    error_m: float | None
    """Straight-line floor distance between measured and projected, or None
    if the point couldn't be projected (see `note`)."""
    note: str = ""

    @property
    def range_m(self) -> float:
        """True distance from the robot's origin."""
        return math.hypot(self.x_true, self.y_true)


def read_reference_points(path: Path) -> list[dict[str, str]]:
    """The CSV's rows, with names filled in and required columns checked."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    for i, row in enumerate(rows, start=2):  # line 1 is the header
        row.setdefault("name", "")
        row["name"] = row["name"] or f"row {i}"
        for column in ("x_m", "y_m"):
            if not row.get(column):
                raise ValueError(f"{path}, line {i}: missing {column}")
        has_pixel = bool(row.get("u")) and bool(row.get("v"))
        if not has_pixel and not row.get("image"):
            raise ValueError(f"{path}, line {i}: needs u and v (pixel mode) or image")
    return rows


def _result(row, prediction, note="") -> PointResult:
    x_true, y_true = float(row["x_m"]), float(row["y_m"])
    if prediction is None:
        return PointResult(row["name"], x_true, y_true, None, None, None, note)
    x_pred, y_pred = prediction
    error = math.hypot(x_pred - x_true, y_pred - y_true)
    return PointResult(row["name"], x_true, y_true, x_pred, y_pred, error, note)


def evaluate_pixel_point(row, params, projection) -> PointResult:
    """Project the row's (u, v) and compare with its measured position."""
    height = float(row.get("height_m") or 0.0)
    try:
        pos = projection.pixel_to_ground(float(row["u"]), float(row["v"]), params, height)
    except projection.ProjectionError as e:
        return _result(row, None, f"not projectable: {e}")
    return _result(row, (pos.x, pos.y))


def evaluate_detection_point(row, params, projection, detector, image_root: Path) -> PointResult:
    """Detect the object in the row's photo, project it, and compare."""
    import cv2  # only needed in detection mode

    image_path = image_root / row["image"]
    image = cv2.imread(str(image_path))
    if image is None:
        return _result(row, None, f"could not read {image_path}")
    class_name = row.get("class_name") or BALL
    candidates = [d for d in detector.predict(image) if d.class_name == class_name]
    if not candidates:
        return _result(row, None, f"no {class_name} detected")
    best = max(candidates, key=lambda d: d.confidence)
    try:
        obj = projection.project_to_ground_plane(best, params)
    except projection.ProjectionError as e:
        return _result(row, None, f"not projectable: {e}")
    return _result(row, (obj.position.x, obj.position.y), f"confidence {best.confidence:.2f}")


def evaluate(rows, params, projection, detector=None, image_root: Path = Path(".")):
    """A PointResult per row, in order. Detection-mode rows need `detector`."""
    results = []
    for row in rows:
        if row.get("u") and row.get("v"):
            results.append(evaluate_pixel_point(row, params, projection))
        elif detector is None:
            results.append(_result(row, None, "detection-mode row, but no --weights given"))
        else:
            results.append(evaluate_detection_point(row, params, projection, detector, image_root))
    return results


def _stats(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"count": 0}
    return {
        "count": len(errors),
        "mean_m": statistics.fmean(errors),
        "median_m": statistics.median(errors),
        "max_m": max(errors),
    }


def summarize(
    results: list[PointResult],
    target_error_m: float = TARGET_ERROR_M,
    target_range_m: float = TARGET_RANGE_M,
    target_band_m: float = TARGET_BAND_M,
) -> dict[str, Any]:
    """Overall and per-distance-band error, and the milestone check.

    The check passes when every projected point within `target_band_m` of
    `target_range_m` is within `target_error_m` of its measured position,
    and none of those points failed to project. It can't pass with no
    points near the target range.
    """
    projected = [r for r in results if r.error_m is not None]
    bands = {}
    for low, high in DISTANCE_BANDS_M:
        label = f"{low:g}-{high:g} m" if high != math.inf else f"{low:g}+ m"
        bands[label] = _stats([r.error_m for r in projected if low <= r.range_m < high])

    near_target = [r for r in results if abs(r.range_m - target_range_m) <= target_band_m]
    near_errors = [r.error_m for r in near_target if r.error_m is not None]
    unprojected_near = [r.name for r in near_target if r.error_m is None]
    target = f"{target_range_m:g} m"
    if not near_target:
        verdict = "no_data"
        reason = f"no reference points within {target_band_m:g} m of {target}"
    elif unprojected_near:
        verdict = "fail"
        reason = f"could not project: {', '.join(unprojected_near)}"
    else:
        worst = max(near_errors)
        verdict = "pass" if worst <= target_error_m else "fail"
        comparison = "<=" if verdict == "pass" else ">"
        reason = (
            f"max error {worst:.3f} m {comparison} {target_error_m:g} m "
            f"over {len(near_errors)} points near {target}"
        )

    return {
        "overall": _stats([r.error_m for r in projected]),
        "unprojected": [{"name": r.name, "note": r.note} for r in results if r.error_m is None],
        "by_distance": bands,
        "target": {
            "error_m": target_error_m,
            "range_m": target_range_m,
            "band_m": target_band_m,
            "verdict": verdict,
            "reason": reason,
        },
    }


def format_report(results: list[PointResult], summary: dict[str, Any]) -> str:
    lines = [f"{'point':<20} {'measured':>15} {'projected':>15} {'range':>7} {'error':>7}"]
    for r in results:
        measured = f"({r.x_true:.2f}, {r.y_true:+.2f})"
        if r.error_m is None:
            lines.append(f"{r.name:<20} {measured:>15} {'—':>15} {r.range_m:>6.2f}m   {r.note}")
            continue
        projected = f"({r.x_pred:.2f}, {r.y_pred:+.2f})"
        lines.append(
            f"{r.name:<20} {measured:>15} {projected:>15} {r.range_m:>6.2f}m {r.error_m:>6.3f}m"
            + (f"  {r.note}" if r.note else "")
        )
    lines.append("")
    lines.append("Error by true distance:")
    for band, stats in summary["by_distance"].items():
        if stats["count"]:
            lines.append(
                f"  {band:<8} n={stats['count']:<3} mean {stats['mean_m']:.3f} m, "
                f"max {stats['max_m']:.3f} m"
            )
    target = summary["target"]
    lines.append("")
    lines.append(
        f"Milestone (<= {target['error_m']:g} m at {target['range_m']:g} m): "
        f"{target['verdict'].upper()} - {target['reason']}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument(
        "--params", type=Path, required=True, help="calibrate.py YAML with a mount: section"
    )
    parser.add_argument("--weights", help="Detector weights, needed for detection-mode rows")
    parser.add_argument("--conf", type=float, default=0.20, help="Detector confidence cutoff")
    parser.add_argument("--target-error", type=float, default=TARGET_ERROR_M)
    parser.add_argument("--target-range", type=float, default=TARGET_RANGE_M)
    parser.add_argument("--target-band", type=float, default=TARGET_BAND_M)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    projection = load_projection()
    params = projection.load_camera_params(args.params)
    rows = read_reference_points(args.reference_csv)

    detector = None
    if args.weights:
        sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))
        from detector import Detector

        detector = Detector(args.weights, conf=args.conf)

    results = evaluate(rows, params, projection, detector, args.reference_csv.parent)
    summary = summarize(results, args.target_error, args.target_range, args.target_band)
    print(format_report(results, summary))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "reference_csv": str(args.reference_csv),
        "params": str(args.params),
        "mount": asdict(params.mount),
        "weights": args.weights,
        "points": [asdict(r) | {"range_m": r.range_m} for r in results],
        **summary,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to {args.output}")
    return 0 if summary["target"]["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
