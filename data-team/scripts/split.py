"""Build data-team's reproducible train/val/test split of TORSO-21 (YOLO format).

Oct Week 4. Produces the dataset described in "Dataset contract" in
detection-team/README.md, so detection-team can switch from their temporary
placeholder to this with a one-line config change.

    python data-team/scripts/split.py                         # ball only -> datasets/ball/
    python data-team/scripts/split.py --classes ball robot goalpost landmark
                                                              # -> datasets/multiclass/

Input: the real-world TORSO-21 download from download.py,
data-team/datasets/TORSO-21/reality/{train,test}/ (change with --torso-dir).

How the split is made (details: data-team/docs/splits-and-augmentation.md):

  - TORSO-21's train/ and test/ are pooled. Their test/ is NOT unseen
    footage: every recording in it also appears in train/.
  - The pool is split by *recording* (the number before the first "-" in a
    file name, e.g. 1028 in 1028-img00155.png), never by single image, so
    every frame of a recording lands in the same split. Frames of one
    recording are near-duplicates; splitting by image would let the model
    be tested on copies of its training frames.
  - Recordings are shuffled with a fixed seed and assigned by their number
    of *ball* boxes, whatever --classes is, so ball-only and multi-class
    datasets test on exactly the same recordings.
  - Each split keeps every image with an object of --classes, plus some
    images annotators confirmed contain none of them (negatives).

Same download + same --seed => identical split. split_info.yaml lists every
recording in every split and a fingerprint to compare with teammates.

Classes (ids = positions in shared/classes.py's CLASSES), via
explore_torso21.TORSO_TO_OURS:
    ball, robot  <- TORSO-21 boxes
    goalpost     <- 4-point polygon -> its bounding box
    landmark     <- L/T/X intersections (points) -> square, --landmark-size of
                    the image width, centered on the point
    field_line   not produced (team decision: lines handled another way later)

Output (gitignored):

    data.yaml                      # path/train/val/test/names, for ultralytics
    split_info.yaml                # seed, settings, recordings per split, counts
    images/{train,val,test}/       # symlinks to TORSO-21 images (--copy to copy)
    labels/{train,val,test}/*.txt  # "<class id> cx cy w h", normalized; empty = no objects
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from explore_torso21 import TORSO_TO_OURS  # noqa: E402
from yolo_io import write_labels  # noqa: E402

from shared.classes import BALL, CLASSES, GOALPOST, LANDMARK, ROBOT  # noqa: E402

# Team seed for every split we hand to detection-team. Change it only if the
# whole team agrees: scores are only comparable on the same split.
DEFAULT_SEED = 42

DATASETS_DIR = REPO_ROOT / "data-team" / "datasets"
DEFAULT_TORSO_DIR = DATASETS_DIR / "TORSO-21"
SPLITS = ("train", "val", "test")
SUPPORTED_CLASSES = (BALL, ROBOT, GOALPOST, LANDMARK)  # field_line: no boxes, by decision
DEFAULT_LANDMARK_SIZE = 0.05  # landmark box side, as a share of image width
MIN_BOX_PIXELS = 2


def recording_of(name: str) -> str:
    """Recording an image came from: the number before the first '-' in its name."""
    return name.split("-", 1)[0]


def load_torso(torso_dir: Path) -> dict:
    """All real-world TORSO-21 images (train/ + test/ pooled), keyed by file name,
    each with an added "image_path"."""
    images = {}
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # C loader is much faster
    for half in ("train", "test"):
        root = torso_dir / "reality" / half
        if not (root / "annotations.yaml").exists():
            sys.exit(
                f"No TORSO-21 real-world {half} set at {root}. Download it with:\n"
                "  python data-team/scripts/download.py --subset reality"
            )
        print(f"Loading {root / 'annotations.yaml'} ...")
        with (root / "annotations.yaml").open() as f:
            entries = yaml.load(f, Loader=loader)["images"]
        for name, entry in entries.items():
            if name in images:
                continue  # same file in both halves: keep one
            entry["image_path"] = root / "images" / name
            images[name] = entry
    return images


def object_boxes(image: dict, classes, landmark_size: float = DEFAULT_LANDMARK_SIZE) -> list:
    """[(class name, (x_min, y_min, x_max, y_max) in pixels)] for every visible
    object of `classes`, clipped to the image, slivers dropped."""
    width, height = image["width"], image["height"]
    objects = []
    for ann in image.get("annotations") or []:
        ours = TORSO_TO_OURS.get(ann.get("type"))
        if ours not in classes or not ann.get("in_image", True) or not ann.get("vector"):
            continue
        points = ann["vector"]
        if len(points) == 1:  # a point (line intersection): square box around it
            (x, y), half = points[0], landmark_size * width / 2
            xs, ys = (x - half, x + half), (y - half, y + half)
        else:  # a 2-corner box or a polygon
            xs, ys = [p[0] for p in points], [p[1] for p in points]
        x0, x1 = (min(max(v, 0), width) for v in (min(xs), max(xs)))
        y0, y1 = (min(max(v, 0), height) for v in (min(ys), max(ys)))
        if x1 - x0 < MIN_BOX_PIXELS or y1 - y0 < MIN_BOX_PIXELS:
            continue
        objects.append((ours, (float(x0), float(y0), float(x1), float(y1))))
    return objects


def confirmed_empty(image: dict, classes) -> bool:
    """True if annotators marked every wanted class as not in the image.

    Images where a class simply wasn't labeled either way are not used as
    negatives: we can't be sure they are really empty."""
    absent = set()
    for ann in image.get("annotations") or []:
        ours = TORSO_TO_OURS.get(ann.get("type"))
        if ours in classes:
            if ann.get("in_image", True):
                return False
            absent.add(ours)
    return absent == set(classes)


def assign_recordings(
    ball_counts: dict, fractions: dict, seed: int, tolerance=1.15, max_share=0.25
) -> dict:
    """Shuffle recordings (seeded) and walk through them, putting each into
    test, then val, while that split is below its share of all ball boxes.
    Everything else goes to train. Returns {recording: split}.

    Two guards keep val/test a fair sample of "unseen footage":
      - a recording with more than `max_share` (25%) of a split's target
        never goes there. TORSO-21 has a few huge recordings (up to ~1000
        balls); without this, one or two of them can make up the whole test
        set. They go to train instead.
      - a recording that would push a split more than 15% past its target
        is skipped (also goes to train)."""
    recordings = sorted(ball_counts, key=lambda r: (len(r), r))
    random.Random(seed).shuffle(recordings)
    total = sum(ball_counts.values())
    targets = {s: fractions[s] * total for s in ("test", "val")}
    filled = dict.fromkeys(targets, 0)
    assignment = {}
    for rec in recordings:
        assignment[rec] = "train"
        n = ball_counts[rec]
        for split in ("test", "val"):
            too_big = n > max_share * targets[split]
            room = targets[split] * tolerance
            fits = filled[split] < targets[split] and filled[split] + n <= room
            if fits and not too_big:
                assignment[rec] = split
                filled[split] += n
                break
    return assignment


def place_image(src: Path, dst: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build(
    torso_dir: Path,
    out: Path,
    classes: tuple,
    fractions: dict,
    seed: int,
    negatives: float,
    landmark_size: float = DEFAULT_LANDMARK_SIZE,
    copy: bool = False,
) -> dict:
    images = load_torso(torso_dir)
    objects = {n: object_boxes(e, classes, landmark_size) for n, e in images.items()}

    # 1. Recordings -> splits, by ball count (same for every --classes).
    ball_counts = defaultdict(int)
    for name, entry in images.items():
        ball_counts[recording_of(name)] += len(object_boxes(entry, (BALL,)))
    assignment = assign_recordings(ball_counts, fractions, seed)

    # 2. Images per split: all with objects + seeded sample of confirmed-empty ones.
    rng = random.Random(seed)
    by_split = {}
    for split in SPLITS:
        names = sorted(n for n in images if assignment[recording_of(n)] == split)
        positives = [n for n in names if objects[n]]
        empties = [n for n in names if not objects[n] and confirmed_empty(images[n], classes)]
        keep = min(len(empties), round(len(positives) * negatives))
        by_split[split] = sorted(positives + rng.sample(empties, keep))

    empty = [s for s in SPLITS if not by_split[s]]
    if empty:
        sys.exit(
            f"No images ended up in {empty}: every recording is too big for those splits. "
            "Raise --val/--test or use more data."
        )
    recs = {s: {recording_of(n) for n in names} for s, names in by_split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if recs[a] & recs[b]:  # can't happen by construction; never write a leaky split
            sys.exit(f"BUG: recordings in both {a} and {b}: {sorted(recs[a] & recs[b])}")

    # 3. Write it. Start from a clean folder so nothing stale survives.
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    counts = {}
    for split, names in by_split.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        per_class = dict.fromkeys(classes, 0)
        for name in names:
            entry = images[name]
            place_image(entry["image_path"], out / "images" / split / name, copy)
            ids = [CLASSES.index(c) for c, _ in objects[name]]
            boxes = [box for _, box in objects[name]]
            for c, _ in objects[name]:
                per_class[c] += 1
            label = out / "labels" / split / f"{Path(name).stem}.txt"
            write_labels(label, ids, boxes, entry["width"], entry["height"])
        counts[split] = {
            "images": len(names),
            "images_with_objects": sum(1 for n in names if objects[n]),
            "objects": per_class,
            "recordings": sorted(recs[split], key=lambda r: (len(r), r)),
        }

    highest = max(CLASSES.index(c) for c in classes)
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        # ids are positions in shared/classes.py; YOLO needs 0..n-1 all named,
        # so classes in between that we don't label are listed but never used.
        "names": {i: CLASSES[i] for i in range(highest + 1)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    lists = {s: by_split[s] for s in SPLITS}
    fingerprint = hashlib.sha256(json.dumps(lists, sort_keys=True).encode()).hexdigest()[:16]
    info = {
        "source": "TORSO-21 real-world (train + test pooled), data-team/scripts/split.py",
        "classes": list(classes),
        "seed": seed,
        "fractions": fractions,
        "split_by": "recording, assigned by ball count",
        "negatives_per_positive": negatives,
        "landmark_box_fraction": landmark_size if LANDMARK in classes else None,
        "fingerprint": fingerprint,
        "splits": counts,
    }
    (out / "split_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--torso-dir", type=Path, default=DEFAULT_TORSO_DIR)
    parser.add_argument(
        "--classes", nargs="+", default=[BALL], choices=SUPPORTED_CLASSES, help="Default: ball"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Default: datasets/ball or /multiclass"
    )
    parser.add_argument("--val", type=float, default=0.10, help="Share of ball boxes for val")
    parser.add_argument("--test", type=float, default=0.10, help="Share of ball boxes for test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--negatives",
        type=float,
        default=0.1,
        help="Confirmed-empty images per image with objects (default 0.1)",
    )
    parser.add_argument(
        "--landmark-size",
        type=float,
        default=DEFAULT_LANDMARK_SIZE,
        help="Landmark box side as a share of image width (default 0.05)",
    )
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()
    if not 0 < args.val + args.test < 1:
        sys.exit("--val + --test must be between 0 and 1")

    classes = tuple(c for c in CLASSES if c in args.classes)  # keep CLASSES order
    out = args.out or DATASETS_DIR / ("ball" if classes == (BALL,) else "multiclass")
    fractions = {"train": round(1 - args.val - args.test, 6), "val": args.val, "test": args.test}
    info = build(
        args.torso_dir,
        out,
        classes,
        fractions,
        args.seed,
        args.negatives,
        args.landmark_size,
        args.copy,
    )

    print(f"\nWrote {out}  (seed {info['seed']}, fingerprint {info['fingerprint']})")
    for split, c in info["splits"].items():
        objs = ", ".join(f"{n} {name}" for name, n in c["objects"].items())
        print(
            f"{split:<5} {c['images']:>6} images ({c['images_with_objects']} with objects), "
            f"{objs}, {len(c['recordings'])} recordings"
        )
    print("No recording appears in more than one split.")
    print(f"Train with: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
