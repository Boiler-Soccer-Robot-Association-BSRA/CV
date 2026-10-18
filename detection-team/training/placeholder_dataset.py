"""Build a TEMPORARY detection dataset from TORSO-21, so detection-team can
train and evaluate before data-team's real split exists.

This is the stopgap ARCHITECTURE.md's cross-team handoffs #1, #4 and #5
call for. It is deliberately narrow (no augmentation, no relabeling) and
is meant to be thrown away once data-team/scripts/split.py produces the
real dataset in the same layout (see "Dataset contract" in
detection-team/README.md).

    python detection-team/training/placeholder_dataset.py                 # ball only
    python detection-team/training/placeholder_dataset.py --classes ball robot goalpost landmark

How TORSO-21 labels become our classes (shared/classes.py):

    ball      <- ball (box)
    robot     <- robot (box)
    goalpost  <- goalpost (4-point polygon -> its bounding box)
    landmark  <- L-, T- and X-Intersection (a single point -> a square
                 LANDMARK_BOX_FRACTION of the image width, centered on it)
    field_line   not produced: TORSO-21 only has pixel masks for lines, and
                 the team decided field lines stay out of the box detector

Reads the real-world TORSO-21 download made by
data-team/scripts/download.py (override the location with --torso-dir):

    data-team/datasets/TORSO-21/reality/{train,test}/{annotations.yaml,images/}

(The layout you get from unzipping train.zip / test.zip by hand,
TORSO-21/reality_train/train/ and reality_test/test/, works too.)

and writes a YOLO-format dataset (images are symlinked, not copied) to
detection-team/datasets/placeholder_ball/ or placeholder_multiclass/
(gitignored):

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
Both halves are pooled and re-split. Recordings are always assigned by
their *ball* count with the same seed, whatever --classes is, so ball and
multi-class models are tested on exactly the same recordings and their
ball scores can be compared.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data-team" / "scripts"))

from shared.classes import BALL, CLASSES, GOALPOST, LANDMARK, ROBOT  # noqa: E402

DEFAULT_TORSO_DIR = REPO_ROOT / "data-team" / "datasets" / "TORSO-21"
DATASETS_DIR = REPO_ROOT / "detection-team" / "datasets"
SPLITS = ("train", "val", "test")
MIN_BOX_PIXELS = 2
# Intersections are labeled as single points; give them a box this
# fraction of the image width so a detector has something to regress.
LANDMARK_BOX_FRACTION = 0.05

# data-team's TORSO-21 -> shared/classes.py mapping is the one source of
# truth (data-team/README.md, "Mapping to our classes").
from explore_torso21 import TORSO_TO_OURS  # noqa: E402

SUPPORTED_CLASSES = (BALL, ROBOT, GOALPOST, LANDMARK)

Box = tuple[float, float, float, float]


def recording_id(image_name: str) -> str:
    return image_name.split("-", 1)[0]


def object_boxes(image: dict, classes: tuple[str, ...] = (BALL,)) -> list[tuple[str, Box]]:
    """(our class name, pixel box (x_min, y_min, x_max, y_max)) for every
    visible object of `classes` in one TORSO-21 image entry, clipped to the
    image and with slivers dropped. See the module docstring for how each
    TORSO-21 label type is turned into a box."""
    width, height = image["width"], image["height"]
    objects = []
    for ann in image.get("annotations", []):
        class_name = TORSO_TO_OURS.get(ann.get("type"))
        if class_name not in classes or not ann.get("in_image"):
            continue
        points = ann["vector"]
        if len(points) == 1:  # a point label (intersection)
            (x, y), half = points[0], LANDMARK_BOX_FRACTION * width / 2
            xs, ys = (x - half, x + half), (y - half, y + half)
        else:  # a 2-corner box or a polygon
            xs, ys = [p[0] for p in points], [p[1] for p in points]
        x_min, x_max = (min(max(v, 0), width) for v in (min(xs), max(xs)))
        y_min, y_max = (min(max(v, 0), height) for v in (min(ys), max(ys)))
        if x_max - x_min < MIN_BOX_PIXELS or y_max - y_min < MIN_BOX_PIXELS:
            continue
        objects.append((class_name, (float(x_min), float(y_min), float(x_max), float(y_max))))
    return objects


def ball_boxes(image: dict) -> list[Box]:
    return [box for _, box in object_boxes(image, (BALL,))]


def yolo_label_lines(objects: list[tuple[str, Box]], width: int, height: int) -> list[str]:
    """YOLO label lines; class ids are positions in shared/classes.py's CLASSES."""
    lines = []
    for class_name, (x_min, y_min, x_max, y_max) in objects:
        class_id = CLASSES.index(class_name)
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
        candidates = [torso_dir / "reality" / half, torso_dir / f"reality_{half}" / half]
        root = next((c for c in candidates if (c / "annotations.yaml").exists()), None)
        if root is None:
            raise SystemExit(
                f"No TORSO-21 real-world {half} set under {torso_dir}. Download it with: "
                "python data-team/scripts/download.py --subset reality"
            )
        annotations = root / "annotations.yaml"
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


def default_out_dir(classes: tuple[str, ...]) -> Path:
    return DATASETS_DIR / ("placeholder_ball" if classes == (BALL,) else "placeholder_multiclass")


def build(
    torso_dir: Path,
    out_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    negatives_per_positive: float,
    classes: tuple[str, ...] = (BALL,),
) -> dict:
    unsupported = set(classes) - set(SUPPORTED_CLASSES)
    if unsupported:
        raise SystemExit(f"Can't build {sorted(unsupported)} from TORSO-21 (see module docstring)")
    images = load_torso(torso_dir)
    objects_by_image = {name: object_boxes(entry, classes) for name, entry in images.items()}

    ball_counts: dict[str, int] = defaultdict(int)
    for name, entry in images.items():
        ball_counts[recording_id(name)] += len(ball_boxes(entry))
    assignment = assign_recordings(ball_counts, val_fraction, test_fraction, seed)

    # Every image with an object, plus a seeded sample of empty images (so
    # the model also learns what "nothing here" looks like).
    rng = random.Random(seed)
    by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        names = sorted(n for n in images if assignment[recording_id(n)] == split)
        positives = [n for n in names if objects_by_image[n]]
        negatives = [n for n in names if not objects_by_image[n]]
        keep = min(len(negatives), round(len(positives) * negatives_per_positive))
        by_split[split] = sorted(positives + rng.sample(negatives, keep))

    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and isn't empty; delete it to rebuild.")

    counts = {}
    for split, names in by_split.items():
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        per_class: dict[str, int] = {c: 0 for c in classes}
        for name in names:
            entry = images[name]
            (image_dir / name).symlink_to(entry["image_path"].resolve())
            objects = objects_by_image[name]
            for class_name, _ in objects:
                per_class[class_name] += 1
            lines = yolo_label_lines(objects, entry["width"], entry["height"])
            (label_dir / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines) + "\n" if lines else ""
            )
        counts[split] = {
            "images": len(names),
            "images_with_objects": sum(1 for n in names if objects_by_image[n]),
            "objects": per_class,
            "recordings": sorted({recording_id(n) for n in names}, key=int),
        }

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        # Class ids are positions in shared/classes.py's CLASSES, so an id
        # means the same class in every dataset and model. YOLO needs ids
        # 0..n-1 named, so classes in between that we don't label (e.g.
        # field_line) are still listed; they just never appear.
        "names": {i: CLASSES[i] for i in range(max(CLASSES.index(c) for c in classes) + 1)},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    info = {
        "source": "TORSO-21 real-world (train + test pooled)",
        "classes": list(classes),
        "landmark_box_fraction": LANDMARK_BOX_FRACTION if LANDMARK in classes else None,
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
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[BALL],
        choices=SUPPORTED_CLASSES,
        help="Which of our classes to label (default: ball).",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Default: datasets/placeholder_<ball|multiclass>."
    )
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

    # Keep CLASSES order regardless of how they were typed.
    classes = tuple(c for c in CLASSES if c in args.classes)
    out = args.out or default_out_dir(classes)
    info = build(
        args.torso_dir, out, args.val, args.test, args.seed, args.negatives_per_positive, classes
    )
    for split, c in info["splits"].items():
        objects = ", ".join(f"{n} {name}" for name, n in c["objects"].items())
        print(
            f"{split:<5} {c['images']:>5} images ({c['images_with_objects']} with objects), "
            f"{objects}, {len(c['recordings'])} recordings"
        )
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
