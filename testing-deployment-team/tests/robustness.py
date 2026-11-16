"""Robustness test: how does a detector hold up when the image gets worse?

Dec Week 10. Runs a model over a test split once per *condition* (darker,
blurred, partly covered, ...), applying the condition to every frame, and
reports recall per class per condition next to the clean result.

    python testing-deployment-team/tests/robustness.py \\
        detection-team/models/multiclass_v2.pt \\
        data-team/datasets/multiclass_r180/data.yaml --conf 0.20 \\
        --count-once 151-:ball

Conditions (all fixed, no randomness in strength, so runs are comparable):

| Group     | Condition         | What is done to every frame                                |
|-----------|-------------------|------------------------------------------------------------|
| lighting  | `dim`             | contrast x0.8, brightness -30 (a duller hall)              |
|           | `dark`            | contrast x0.5, brightness -40, gamma 1.5 (lights mostly off)|
|           | `bright`          | contrast x1.2, brightness +40, gamma 0.7 (washed out)      |
|           | `low_contrast`    | contrast x0.5, brightness +64 (hazy, flat)                 |
|           | `warm`, `cool`    | red/blue gain +-25% (tungsten vs. daylight)                |
| blur      | `motion_5/9/15`   | horizontal motion blur, 5 / 9 / 15 px (camera shake)       |
| occlusion | `occluded_25/50`  | 25% / 50% of every labeled object covered by a gray patch  |
|           |                   | on a random side (seeded)                                  |

Lighting and motion blur reuse data-team's transforms
(data-team/scripts/augment.py: `apply_lighting`, `motion_kernel`).
Occlusion is new here: augment.py has no occlusion.

**How it is scored.** Same matching as the eval harness
(`eval_harness.match_frame_indices`: IoU >= 0.5, same class, most
confident first), against the original labels. Recall per class =
objects found / objects labeled. Precision is in the JSON too.

**`--count-once PREFIX:CLASS`** applies the team's rule for recording 151
(one stationary ball labeled in 113 frames): objects of CLASS in frames
whose file name starts with PREFIX count as one object, found if found in
any frame. Both numbers are reported, as the team decided (Nov 8).

**Limits.** These are synthetic changes to test images, not new footage:
blur is measured in pixels of frames that are 320x240 to 800x600, and a
flat gray patch is a cruder occluder than a robot's leg. Camera angle and
other fields/backgrounds (also in Week 10's list) can't be simulated this
way and need real footage.

Results go to --output as JSON (gitignored) and a Markdown table is printed.
`--examples-dir DIR` also saves a few frames per condition, to see what the
model was shown.
"""

import argparse
import importlib.util
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

import eval_harness  # noqa: E402

RESULTS_DIR = REPO_ROOT / "testing-deployment-team" / "benchmarks" / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "robustness.json"
DEFAULT_CONF = 0.20  # the operating point in detection-team's results
OCCLUDER_GRAY = 114
SIDES = ("left", "right", "top", "bottom")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


augment = _load("augment", REPO_ROOT / "data-team" / "scripts" / "augment.py")


# --- conditions -------------------------------------------------------------------
# A condition is a function (image, boxes, rng) -> image. `boxes` are the frame's
# ground-truth boxes as [x_min, y_min, x_max, y_max] lists; only occlusion uses them.

Transform = Callable[[np.ndarray, list[list[float]], random.Random], np.ndarray]


def lighting(**settings: float) -> Transform:
    return lambda img, boxes, rng: augment.apply_lighting(img, **settings)


def motion_blur(length: int, angle_deg: float = 0.0) -> Transform:
    kernel = augment.motion_kernel(length, angle_deg)
    return lambda img, boxes, rng: cv2.filter2D(img, -1, kernel)


def occluder_rect(
    box: list[float], fraction: float, side: str
) -> tuple[float, float, float, float]:
    """The part of `box` to cover: `fraction` of its area, starting from `side`."""
    x_min, y_min, x_max, y_max = box
    width, height = x_max - x_min, y_max - y_min
    if side == "left":
        return x_min, y_min, x_min + width * fraction, y_max
    if side == "right":
        return x_max - width * fraction, y_min, x_max, y_max
    if side == "top":
        return x_min, y_min, x_max, y_min + height * fraction
    if side == "bottom":
        return x_min, y_max - height * fraction, x_max, y_max
    raise ValueError(f"Unknown side {side!r}")


def occlude(fraction: float) -> Transform:
    """Cover `fraction` of every labeled object with a flat gray patch on a
    random side. Patch edges are rounded to whole pixels, so on a 10 px
    ball "25%" is 2 or 3 pixels."""

    def transform(img: np.ndarray, boxes: list[list[float]], rng: random.Random) -> np.ndarray:
        out = img.copy()
        height, width = out.shape[:2]
        for box in boxes:
            x0, y0, x1, y1 = occluder_rect(box, fraction, rng.choice(SIDES))
            x0, x1 = (int(np.clip(round(v), 0, width)) for v in (x0, x1))
            y0, y1 = (int(np.clip(round(v), 0, height)) for v in (y0, y1))
            out[y0:y1, x0:x1] = OCCLUDER_GRAY
        return out

    return transform


@dataclass(frozen=True)
class Condition:
    name: str
    group: str
    description: str
    transform: Transform | None  # None = the clean image


CONDITIONS = [
    Condition("clean", "baseline", "unchanged test images", None),
    Condition(
        "dim", "lighting", "contrast x0.8, brightness -30", lighting(contrast=0.8, brightness=-30)
    ),
    Condition(
        "dark",
        "lighting",
        "contrast x0.5, brightness -40, gamma 1.5",
        lighting(contrast=0.5, brightness=-40, gamma=1.5),
    ),
    Condition(
        "bright",
        "lighting",
        "contrast x1.2, brightness +40, gamma 0.7",
        lighting(contrast=1.2, brightness=40, gamma=0.7),
    ),
    Condition(
        "low_contrast",
        "lighting",
        "contrast x0.5, brightness +64",
        lighting(contrast=0.5, brightness=64),
    ),
    Condition("warm", "lighting", "red +25%, blue -25%", lighting(warmth=0.25)),
    Condition("cool", "lighting", "blue +25%, red -25%", lighting(warmth=-0.25)),
    Condition("motion_5", "blur", "horizontal motion blur, 5 px", motion_blur(5)),
    Condition("motion_9", "blur", "horizontal motion blur, 9 px", motion_blur(9)),
    Condition("motion_15", "blur", "horizontal motion blur, 15 px", motion_blur(15)),
    Condition("occluded_25", "occlusion", "25% of each object covered", occlude(0.25)),
    Condition("occluded_50", "occlusion", "50% of each object covered", occlude(0.50)),
]
CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}


# --- scoring ----------------------------------------------------------------------


def parse_count_once(spec: str) -> tuple[str, str]:
    prefix, sep, class_name = spec.partition(":")
    if not sep or not prefix or not class_name:
        raise argparse.ArgumentTypeError(f"expected PREFIX:CLASS (e.g. 151-:ball), got {spec!r}")
    return prefix, class_name


def evaluate_condition(
    manifest: list[dict],
    manifest_dir: Path,
    detector: Any,
    condition: Condition,
    iou_threshold: float = 0.5,
    seed: int = 0,
    count_once: tuple[str, str] | None = None,
    examples_dir: Path | None = None,
    num_examples: int = 0,
) -> dict[str, Any]:
    """Run `detector` over every frame with `condition` applied and count,
    per class, labeled objects, found objects and false alarms.

    The seed is per condition and per frame, so a frame gets the same
    occluder sides whichever conditions or frames are run with it.
    """
    labeled: dict[str, int] = {}
    found: dict[str, int] = {}
    false_alarms: dict[str, int] = {}
    once_labeled = once_found = 0  # the --count-once group, as scored
    example_step = max(1, len(manifest) // num_examples) if num_examples else 0

    for index, entry in enumerate(manifest):
        image_path = manifest_dir / entry["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        ground_truth = entry.get("annotations", [])
        if condition.transform is not None:
            rng = random.Random(f"{seed}:{image_path.name}")
            image = condition.transform(image, [gt["bbox"] for gt in ground_truth], rng)
        if examples_dir is not None and example_step and index % example_step == 0:
            examples_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(examples_dir / f"{condition.name}__{image_path.stem}.jpg"), image)

        predictions = detector.predict(image)
        matches, missed = eval_harness.match_frame_indices(predictions, ground_truth, iou_threshold)
        in_once_group = count_once is not None and image_path.name.startswith(count_once[0])
        for gt_index, gt in enumerate(ground_truth):
            name = gt["class_name"]
            was_found = gt_index not in missed
            labeled[name] = labeled.get(name, 0) + 1
            found[name] = found.get(name, 0) + was_found
            if in_once_group and name == count_once[1]:
                once_labeled += 1
                once_found += was_found
        for prediction, gt_index in matches:
            if gt_index is None:
                name = prediction.class_name
                false_alarms[name] = false_alarms.get(name, 0) + 1

    per_class = {}
    for name in sorted(labeled):
        fp = false_alarms.get(name, 0)
        per_class[name] = {
            "labeled": labeled[name],
            "found": found[name],
            "false_alarms": fp,
            "recall": found[name] / labeled[name],
            "precision": found[name] / (found[name] + fp) if found[name] + fp else 1.0,
        }
    total_labeled, total_found = sum(labeled.values()), sum(found.values())
    total_fp = sum(false_alarms.values())
    result: dict[str, Any] = {
        "condition": condition.name,
        "group": condition.group,
        "description": condition.description,
        "per_class": per_class,
        "recall": total_found / total_labeled if total_labeled else 1.0,
        "precision": total_found / (total_found + total_fp) if total_found + total_fp else 1.0,
        "labeled": total_labeled,
        "found": total_found,
        "false_alarms": total_fp,
    }
    if count_once is not None and once_labeled:
        # The group's objects become one object, found if any of them was.
        class_name = count_once[1]
        stats = per_class[class_name]
        once_total = stats["labeled"] - once_labeled + 1
        once_hits = stats["found"] - once_found + (1 if once_found else 0)
        result["count_once"] = {
            "prefix": count_once[0],
            "class_name": class_name,
            "group_labeled": once_labeled,
            "group_found": once_found,
            "labeled": once_total,
            "found": once_hits,
            "recall": once_hits / once_total,
        }
    return result


def format_table(results: list[dict[str, Any]]) -> str:
    """Markdown: one row per condition, recall per class, with the change
    from the clean row in points (hundredths)."""
    classes = sorted({name for r in results for name in r["per_class"]})
    clean = next((r for r in results if r["condition"] == "clean"), None)
    once = [r for r in results if "count_once" in r]
    headers = ["Condition", "What changes", *classes, "all"]
    if once:
        headers.insert(-1, f"{once[0]['count_once']['class_name']} (counted once)")

    def cell(value: float, baseline: float | None) -> str:
        if baseline is None:
            return f"{value:.3f}"
        return f"{value:.3f} ({(value - baseline) * 100:+.1f})"

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in results:
        base = clean if clean is not None and r is not clean else None
        row = [f"`{r['condition']}`", r["description"]]
        for name in classes:
            stats = r["per_class"].get(name)
            baseline = base["per_class"][name]["recall"] if base and stats else None
            row.append(cell(stats["recall"], baseline) if stats else "n/a")
        if once:
            value = r.get("count_once")
            baseline = base["count_once"]["recall"] if base and value else None
            row.append(cell(value["recall"], baseline) if value else "n/a")
        row.append(cell(r["recall"], base["recall"] if base else None))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path, help="Model weights (.pt or .onnx).")
    parser.add_argument("dataset", type=Path, help="A YOLO-format data.yaml.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[c.name for c in CONDITIONS],
        choices=[c.name for c in CONDITIONS],
        metavar="NAME",
        help="Which conditions to run (default: all). 'clean' is the baseline.",
    )
    parser.add_argument("--seed", type=int, default=0, help="For the occluders' random sides.")
    parser.add_argument(
        "--count-once",
        type=parse_count_once,
        default=None,
        metavar="PREFIX:CLASS",
        help="Also report CLASS with frames named PREFIX* counted as one object (151-:ball).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only the first N frames (trial).")
    parser.add_argument("--examples-dir", type=Path, default=None, help="Save sample frames here.")
    parser.add_argument("--examples", type=int, default=6, help="Sample frames per condition.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from detector import Detector

    detection_eval = _load("detection_eval", REPO_ROOT / "detection-team" / "training" / "eval.py")
    manifest = detection_eval.yolo_split_to_manifest(args.dataset, args.split)
    if args.limit:
        manifest = manifest[: args.limit]
    detector = Detector(args.weights, conf=args.conf, device=args.device)

    results = []
    for name in args.conditions:
        result = evaluate_condition(
            manifest,
            Path("/"),  # the manifest's image paths are absolute
            detector,
            CONDITIONS_BY_NAME[name],
            args.iou_threshold,
            args.seed,
            args.count_once,
            args.examples_dir,
            args.examples if args.examples_dir else 0,
        )
        results.append(result)
        print(
            f"{name}: recall {result['recall']:.3f}, precision {result['precision']:.3f}",
            flush=True,
        )

    print()
    print(
        f"{args.weights.name} on {args.dataset} [{args.split}], {len(manifest)} frames, "
        f"conf {args.conf}, IoU {args.iou_threshold}, {args.device}. Recall "
        f"(change from clean, in points):"
    )
    print(format_table(results))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(args.weights),
        "dataset": str(args.dataset),
        "split": args.split,
        "num_frames": len(manifest),
        "conf": args.conf,
        "iou_threshold": args.iou_threshold,
        "device": args.device,
        "seed": args.seed,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
