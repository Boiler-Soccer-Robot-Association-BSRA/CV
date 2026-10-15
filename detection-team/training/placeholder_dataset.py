"""Build a TEMPORARY ball-detection dataset from TORSO-21, so detection-team
can train and evaluate before data-team's real split exists.

This is the stopgap ARCHITECTURE.md's cross-team handoff #1 calls for. It
is deliberately narrow (ball only, no augmentation, no relabeling) and is
meant to be thrown away once data-team/scripts/split.py produces the real
dataset in the same layout (see "Dataset contract" in
detection-team/README.md).

    python detection-team/training/placeholder_dataset.py

Reads the real-world TORSO-21 download, expected at (override with
--torso-dir):

    data-team/datasets/TORSO-21/reality_train/train/{annotations.yaml,images/}
    data-team/datasets/TORSO-21/reality_test/test/{annotations.yaml,images/}

and writes a YOLO-format dataset (images are symlinked, not copied) to
detection-team/datasets/placeholder_ball/ (gitignored):

    data.yaml               # what training/train.py and ultralytics read
    split_info.yaml         # seed, which recordings went where, counts
    images/{train,val,test}/...
    labels/{train,val,test}/<image stem>.txt   # "<class id> cx cy w h", normalized

Splits are made by *recording* (the number before the first "-" in a
TORSO-21 filename), not by image: frames from one recording are
near-duplicates, so splitting by image would put copies of training
frames in val/test and overstate how well the model does on footage it
hasn't seen. For the same reason TORSO-21's own train/test division
isn't used — every one of its test recordings also appears in train.
Both halves are pooled and re-split.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.classes import BALL, CLASS_TO_ID  # noqa: E402

DEFAULT_TORSO_DIR = REPO_ROOT / "data-team" / "datasets" / "TORSO-21"
DEFAULT_OUT_DIR = REPO_ROOT / "detection-team" / "datasets" / "placeholder_ball"
SPLITS = ("train", "val", "test")
MIN_BOX_PIXELS = 2


def recording_id(image_name: str) -> str:
    return image_name.split("-", 1)[0]


def ball_boxes(image: dict) -> list[tuple[float, float, float, float]]:
    """Pixel boxes (x_min, y_min, x_max, y_max) of the visible balls in one
    TORSO-21 image entry, clipped to the image and with slivers dropped."""
    width, height = image["width"], image["height"]
    boxes = []
    for ann in image.get("annotations", []):
        if ann.get("type") != "ball" or not ann.get("in_image"):
            continue
        (x1, y1), (x2, y2) = ann["vector"]
        x_min, x_max = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
        y_min, y_max = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
        if x_max - x_min < MIN_BOX_PIXELS or y_max - y_min < MIN_BOX_PIXELS:
            continue
        boxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
    return boxes


def yolo_label_lines(
    boxes: list[tuple[float, float, float, float]], width: int, height: int, class_id: int
) -> list[str]:
    lines = []
    for x_min, y_min, x_max, y_max in boxes:
        cx = (x_min + x_max) / 2 / width
        cy = (y_min + y_max) / 2 / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def load_torso(torso_dir: Path) -> dict[str, dict]:
    """All real-world TORSO-21 images, keyed by filename, each with an
    added "image_path" pointing at the file on disk."""
    images = {}
    for half in ("train", "test"):
        root = torso_dir / f"reality_{half}" / half
        annotations = root / "annotations.yaml"
        if not annotations.exists():
            raise SystemExit(
                f"Missing {annotations}. Download the real-world TORSO-21 set from "
                "https://data.bit-bots.de/TORSO-21/reality/ (train.zip and test.zip) and "
                f"unzip them into {torso_dir}/reality_train and {torso_dir}/reality_test."
            )
        with annotations.open() as f:
            entries = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["images"]
        for name, entry in entries.items():
            entry["image_path"] = root / "images" / name
            images[name] = entry
    return images


def assign_recordings(
    ball_counts: dict[str, int], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    """Shuffle recordings (seeded) and fill test, then val, until each holds
    at least its fraction of all ball boxes; the rest are train. Returns
    {recording id: split}."""
    recordings = sorted(ball_counts)
    random.Random(seed).shuffle(recordings)
    total = sum(ball_counts.values())
    targets = [("test", test_fraction * total), ("val", val_fraction * total)]

    assignment = {}
    filled = 0.0
    for recording in recordings:
        while targets and filled >= targets[0][1]:
            targets.pop(0)
            filled = 0.0
        if targets:
            assignment[recording] = targets[0][0]
            filled += ball_counts[recording]
        else:
            assignment[recording] = "train"
    return assignment


def build(
    torso_dir: Path,
    out_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    negatives_per_positive: float,
) -> dict:
    images = load_torso(torso_dir)
    boxes_by_image = {name: ball_boxes(entry) for name, entry in images.items()}

    ball_counts: dict[str, int] = defaultdict(int)
    for name, boxes in boxes_by_image.items():
        ball_counts[recording_id(name)] += len(boxes)
    assignment = assign_recordings(ball_counts, val_fraction, test_fraction, seed)

    # Every image with a ball, plus a seeded sample of ball-free images
    # (so the model also learns what "no ball" looks like).
    rng = random.Random(seed)
    by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        names = sorted(n for n in images if assignment[recording_id(n)] == split)
        positives = [n for n in names if boxes_by_image[n]]
        negatives = [n for n in names if not boxes_by_image[n]]
        keep = min(len(negatives), round(len(positives) * negatives_per_positive))
        by_split[split] = sorted(positives + rng.sample(negatives, keep))

    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and isn't empty; delete it to rebuild.")

    class_id = CLASS_TO_ID[BALL]
    counts = {}
    for split, names in by_split.items():
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        num_boxes = 0
        for name in names:
            entry = images[name]
            (image_dir / name).symlink_to(entry["image_path"].resolve())
            lines = yolo_label_lines(
                boxes_by_image[name], entry["width"], entry["height"], class_id
            )
            num_boxes += len(lines)
            (label_dir / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines) + "\n" if lines else ""
            )
        counts[split] = {
            "images": len(names),
            "images_with_ball": sum(1 for n in names if boxes_by_image[n]),
            "balls": num_boxes,
            "recordings": sorted({recording_id(n) for n in names}, key=int),
        }

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        # Class ids follow shared/classes.py, so ball stays id 0 when more
        # classes are added later.
        "names": {class_id: BALL},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    info = {
        "source": "TORSO-21 real-world (train + test pooled), ball only",
        "temporary": "placeholder until data-team's split exists; see placeholder_dataset.py",
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "negatives_per_positive": negatives_per_positive,
        "splits": counts,
    }
    (out_dir / "split_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--torso-dir", type=Path, default=DEFAULT_TORSO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--val", type=float, default=0.10, help="Fraction of balls for val.")
    parser.add_argument("--test", type=float, default=0.15, help="Fraction of balls for test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--negatives-per-positive",
        type=float,
        default=0.5,
        help="Ball-free images to include per image with a ball (default 0.5).",
    )
    args = parser.parse_args()

    info = build(
        args.torso_dir, args.out, args.val, args.test, args.seed, args.negatives_per_positive
    )
    for split, c in info["splits"].items():
        print(
            f"{split:<5} {c['images']:>5} images ({c['images_with_ball']} with a ball), "
            f"{c['balls']} balls, {len(c['recordings'])} recordings"
        )
    print(f"Wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
