"""Catalog and visualize a TORSO-21 split (Sept Week 2-3 exploration).

Point it at one split folder, i.e. the one that contains
`annotations.yaml`, `images/` and (usually) `segmentations/`:

    python data-team/scripts/explore_torso21.py data-team/datasets/TORSO-21/reality/test
    python data-team/scripts/explore_torso21.py <split_dir> --viz 12

It prints:
  - image count, resolutions, leagues and recording locations
  - every annotation `type` with present / not-in-image counts, how many
    points its `vector` has, and how often it is blurred / concealed
  - robot team colors and player numbers
  - the distinct colors used in a sample of segmentation masks
  - how the TORSO-21 types line up with our classes (shared/classes.py)

With --viz N it also uses OpenCV to draw the labels on N random images
(boxes, goalpost polygons, intersection points, segmentation overlay) and
writes them to <split_dir>/../viz_<split>/ (inside datasets/, which is
gitignored).

The schema this relies on is documented in data-team/README.md.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.classes import BALL, CLASSES, FIELD_LINE, GOALPOST, LANDMARK, ROBOT

# TORSO-21 `type` -> our class. None = deliberately not used (yet).
# Keep in sync with the "Mapping to our classes" table in data-team/README.md.
TORSO_TO_OURS = {
    "ball": BALL,
    "robot": ROBOT,
    "goalpost": GOALPOST,
    "L-Intersection": LANDMARK,
    "T-Intersection": LANDMARK,
    "X-Intersection": LANDMARK,
    "obstacle": None,
}
# field_line has no per-object label in TORSO-21: it only exists as pixels
# in the segmentation masks.
FROM_SEGMENTATION = {FIELD_LINE}

COLORS = {  # BGR, for drawing only
    "ball": (0, 165, 255),
    "robot": (255, 0, 255),
    "goalpost": (255, 255, 0),
    "obstacle": (128, 128, 128),
    "L-Intersection": (0, 0, 255),
    "T-Intersection": (0, 255, 255),
    "X-Intersection": (255, 0, 0),
}


def load_annotations(split_dir: Path) -> dict:
    path = split_dir / "annotations.yaml"
    if not path.exists():
        sys.exit(f"No annotations.yaml in {split_dir}")
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # C loader is much faster
    print(f"Loading {path} ...")
    with path.open() as f:
        data = yaml.load(f, Loader=loader)
    return data["images"]


def catalog(images: dict) -> None:
    resolutions = Counter()
    leagues = Counter()
    locations = Counter()
    present = Counter()
    absent = Counter()
    n_points = defaultdict(Counter)
    blurred = Counter()
    concealed = Counter()
    robot_color = Counter()
    robot_number = Counter()
    per_image = defaultdict(list)  # type -> objects per image (present only)

    for info in images.values():
        resolutions[(info.get("width"), info.get("height"))] += 1
        meta = info.get("metadata") or {}
        leagues[meta.get("League", meta.get("league", "?"))] += 1
        locations[meta.get("location", "?")] += 1
        counts = Counter()
        for ann in info.get("annotations") or []:
            t = ann["type"]
            if not ann.get("in_image", True):
                absent[t] += 1
                continue
            present[t] += 1
            counts[t] += 1
            n_points[t][len(ann.get("vector") or [])] += 1
            blurred[t] += bool(ann.get("blurred"))
            concealed[t] += bool(ann.get("concealed"))
            if t == "robot":
                robot_color[ann.get("color")] += 1
                robot_number[ann.get("number")] += 1
        for t in present:
            per_image[t].append(counts[t])

    n = len(images)
    print(f"\n== {n} images ==")
    print("Resolutions (w x h):")
    for (w, h), c in resolutions.most_common():
        print(f"  {w} x {h}: {c}")
    print("Leagues:", dict(leagues.most_common()))
    print(f"Locations: {len(locations)} distinct; top 10:")
    for loc, c in locations.most_common(10):
        print(f"  {c:6d}  {loc}")

    print("\n== Annotation types ==")
    header = (
        f"{'type':16s} {'objects':>8s} {'imgs w/':>8s} {'not-in-img':>10s} "
        f"{'blurred':>8s} {'concealed':>9s}  points/vector"
    )
    print(header)
    print("-" * len(header))
    for t in sorted(set(present) | set(absent)):
        imgs_with = sum(1 for c in per_image[t] if c) if t in per_image else 0
        pts = ", ".join(f"{k}pt x{v}" for k, v in sorted(n_points[t].items()))
        print(f"{t:16s} {present[t]:8d} {imgs_with:8d} {absent[t]:10d} "
              f"{blurred[t]:8d} {concealed[t]:9d}  {pts}")

    if robot_color:
        print("\nRobot team colors:", dict(robot_color.most_common()))
        print("Robot numbers:", dict(sorted(robot_number.items(), key=lambda kv: str(kv[0]))))

    print("\n== Mapping to shared/classes.py ==")
    unknown = sorted(set(present) - set(TORSO_TO_OURS))
    for ours in CLASSES:
        sources = [t for t, o in TORSO_TO_OURS.items() if o == ours]
        total = sum(present[t] for t in sources)
        if ours in FROM_SEGMENTATION:
            print(f"  {ours:11s} <- segmentation masks only (no boxes in TORSO-21)")
        elif sources:
            print(f"  {ours:11s} <- {', '.join(sources)}  ({total} objects)")
        else:
            print(f"  {ours:11s} <- NOTHING in TORSO-21")
    ignored = [t for t, o in TORSO_TO_OURS.items() if o is None and present[t]]
    if ignored:
        print("  ignored:", ", ".join(f"{t} ({present[t]})" for t in ignored))
    if unknown:
        print("  WARNING: types not in TORSO_TO_OURS:", ", ".join(unknown))


def mask_path(split_dir: Path, image_name: str) -> Path:
    return split_dir / "segmentations" / (Path(image_name).stem + ".png")


def segmentation_colors(split_dir: Path, names: list, sample: int) -> None:
    seg_dir = split_dir / "segmentations"
    if not seg_dir.is_dir():
        print("\n(no segmentations/ folder)")
        return
    colors = Counter()
    checked = 0
    for name in names[:sample]:
        mask = cv2.imread(str(mask_path(split_dir, name)), cv2.IMREAD_COLOR)
        if mask is None:
            continue
        checked += 1
        uniq, cnt = np.unique(mask.reshape(-1, 3), axis=0, return_counts=True)
        for (b, g, r), c in zip(uniq.tolist(), cnt.tolist()):
            colors[(r, g, b)] += c
    total = sum(colors.values()) or 1
    print(f"\n== Segmentation mask colors (RGB) over {checked} masks ==")
    for rgb, c in colors.most_common(10):
        print(f"  {rgb!s:18s} {100 * c / total:5.1f}% of pixels")


def draw(split_dir: Path, name: str, info: dict) -> np.ndarray:
    img = cv2.imread(str(split_dir / "images" / name), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(split_dir / "images" / name)
    mask = cv2.imread(str(mask_path(split_dir, name)), cv2.IMREAD_COLOR)
    if mask is not None and mask.shape == img.shape:
        img = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    for ann in info.get("annotations") or []:
        if not ann.get("in_image", True):
            continue
        t = ann["type"]
        color = COLORS.get(t, (255, 255, 255))
        pts = np.array(ann["vector"], dtype=np.int32)
        if len(pts) == 1:
            cv2.circle(img, tuple(int(v) for v in pts[0]), 6, color, 2)
        elif len(pts) == 2:
            top_left = tuple(int(v) for v in pts[0])
            bottom_right = tuple(int(v) for v in pts[1])
            cv2.rectangle(img, top_left, bottom_right, color, 2)
        else:
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        label = t
        if ann.get("blurred"):
            label += " (blur)"
        if ann.get("concealed"):
            label += " (conc)"
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("split_dir", type=Path, help="Folder containing annotations.yaml")
    parser.add_argument("--viz", type=int, default=0, metavar="N",
                        help="Draw labels on N random images")
    parser.add_argument("--mask-sample", type=int, default=50, help="Masks to scan for colors")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    images = load_annotations(args.split_dir)
    catalog(images)
    names = sorted(images)
    random.Random(args.seed).shuffle(names)
    segmentation_colors(args.split_dir, names, args.mask_sample)

    if args.viz:
        out = args.split_dir.parent / f"viz_{args.split_dir.name}"
        out.mkdir(exist_ok=True)
        for name in names[: args.viz]:
            cv2.imwrite(str(out / name), draw(args.split_dir, name, images[name]))
        print(f"\nWrote {min(args.viz, len(names))} annotated images to {out}")


if __name__ == "__main__":
    main()
