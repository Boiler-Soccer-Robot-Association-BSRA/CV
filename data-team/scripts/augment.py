"""Offline data augmentation for a YOLO dataset made by split.py.

Oct Week 5. Takes the **train** split (never val/test - those must stay
real so our scores mean something) and writes N augmented copies of each
image, with the YOLO boxes moved to match:

    python data-team/scripts/augment.py data-team/datasets/ball
    python data-team/scripts/augment.py data-team/datasets/ball --copies 2 --preview 12

Augmentations (each copy picks a random combination, with --seed so it's
reproducible):

  lighting    brightness/contrast, gamma, and a warm/cool color cast -
              RoboCup halls mix ceiling lights and window daylight
  blur        Gaussian blur (out of focus) or motion blur in a random
              direction (camera on a walking robot)
  crop/scale  zoom in (crop + resize) or zoom out (shrink + gray padding),
              so the ball shows up at more sizes; boxes that end up mostly
              outside the image are dropped

Output, next to the original split:

    ball/images/train_aug/   ball/labels/train_aug/
    ball/data_aug.yaml       # same as data.yaml but trains on train + train_aug
    ball/augment_info.json   # settings used
    ball/aug_preview/        # with --preview: before/after images with boxes drawn

Train with data_aug.yaml to use the augmented images. (Ultralytics also
does its own on-the-fly augmentation; this script gives us control over the
RoboCup-specific cases and lets us look at exactly what the model sees.)
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_io import read_labels, write_labels

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
PAD_GRAY = 114  # same padding color ultralytics uses


# --- individual augmentations ---------------------------------------------------
# Each takes an image (H x W x 3, uint8, BGR) and a random.Random, and returns
# a new image. crop_scale also moves the boxes.


def adjust_lighting(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random brightness/contrast, gamma and color cast."""
    out = img.astype(np.float32)
    contrast = rng.uniform(0.6, 1.4)
    brightness = rng.uniform(-40, 40)
    out = out * contrast + brightness
    # Per-channel gain: >1 on red = warm (tungsten), >1 on blue = cool (daylight).
    warmth = rng.uniform(-0.12, 0.12)
    out *= np.array([1 - warmth, 1.0, 1 + warmth], dtype=np.float32)  # B, G, R
    out = np.clip(out, 0, 255)
    gamma = rng.uniform(0.6, 1.6)
    out = 255.0 * (out / 255.0) ** gamma
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    return k / max(k.sum(), 1e-6)


def apply_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gaussian blur or motion blur."""
    if rng.random() < 0.5:
        k = rng.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (k, k), 0)
    length = rng.choice([5, 7, 9, 11, 15])
    return cv2.filter2D(img, -1, motion_kernel(length, rng.uniform(0, 180)))


def crop_scale(img, boxes, rng: random.Random, min_visible: float = 0.4, tries: int = 10):
    """Zoom in (random crop, resized back) or zoom out (shrink + pad).

    boxes: (N, 4) pixel xyxy. Returns (new_img, kept_box_indices, new_boxes).
    The output image has the same size as the input.
    """
    h, w = img.shape[:2]
    for _ in range(tries):
        scale = rng.uniform(0.6, 1.3)
        if scale <= 1.0:  # zoom in: crop a window of size scale * (w, h)
            cw, ch = int(w * scale), int(h * scale)
            x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
            out = cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (w, h))
            fx, fy, ox, oy = w / cw, h / ch, -x0 * w / cw, -y0 * h / ch
        else:  # zoom out: shrink the image and place it on a gray canvas
            nw, nh = int(w / scale), int(h / scale)
            ox, oy = rng.randint(0, w - nw), rng.randint(0, h - nh)
            out = np.full_like(img, PAD_GRAY)
            out[oy : oy + nh, ox : ox + nw] = cv2.resize(img, (nw, nh))
            fx, fy = nw / w, nh / h
        moved = boxes * np.array([fx, fy, fx, fy]) + np.array([ox, oy, ox, oy])
        clipped = moved.copy()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, w)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, h)
        area = (moved[:, 2] - moved[:, 0]) * (moved[:, 3] - moved[:, 1])
        visible = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        keep = np.where(
            (visible >= min_visible * np.maximum(area, 1e-6))
            & (clipped[:, 2] - clipped[:, 0] >= 2)
            & (clipped[:, 3] - clipped[:, 1] >= 2)
        )[0]
        # Don't turn a ball image into an (unlabeled-looking) empty one.
        if len(boxes) == 0 or len(keep) > 0:
            return out, keep, clipped[keep]
    return img.copy(), np.arange(len(boxes)), boxes.copy()


def augment(img, boxes, rng: random.Random, p_light=0.8, p_blur=0.3, p_crop=0.5):
    """Apply a random combination; always applies at least one augmentation."""
    keep = np.arange(len(boxes))
    applied = []
    while not applied:
        if rng.random() < p_crop:
            img, keep, boxes = crop_scale(img, boxes, rng)
            applied.append("crop_scale")
        if rng.random() < p_light:
            img = adjust_lighting(img, rng)
            applied.append("lighting")
        if rng.random() < p_blur:
            img = apply_blur(img, rng)
            applied.append("blur")
    return img, keep, boxes, applied


# --- driver ---------------------------------------------------------------------


def draw_boxes(img, boxes, color=(0, 165, 255)):
    out = img.copy()
    for x0, y0, x1, y1 in boxes.astype(int):
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_dir", type=Path, help="Output folder of split.py")
    parser.add_argument("--split", default="train", help="Split to augment (default: train)")
    parser.add_argument("--copies", type=int, default=1, help="Augmented copies per image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-light", type=float, default=0.8, help="Chance of lighting change")
    parser.add_argument("--p-blur", type=float, default=0.3, help="Chance of blur")
    parser.add_argument("--p-crop", type=float, default=0.5, help="Chance of crop/scale")
    parser.add_argument("--preview", type=int, default=0, metavar="N", help="Save N before/after")
    args = parser.parse_args()

    if args.split in ("val", "test"):
        sys.exit("Only augment training data - val/test must stay untouched.")
    root = args.dataset_dir
    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    if not img_dir.is_dir():
        sys.exit(f"{img_dir} not found. Run split.py first.")
    out_name = f"{args.split}_aug"
    out_img, out_lbl = root / "images" / out_name, root / "labels" / out_name
    for d in (out_img, out_lbl):
        if d.exists():
            shutil.rmtree(d)  # start clean so re-runs are reproducible
        d.mkdir(parents=True)
    preview_dir = root / "aug_preview"
    if args.preview:
        shutil.rmtree(preview_dir, ignore_errors=True)
        preview_dir.mkdir()

    rng = random.Random(args.seed)
    names = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    # Preview images that actually have boxes, spread across the whole split,
    # so you can check the boxes still sit on the objects.
    labels = [lbl_dir / (Path(n).stem + ".txt") for n in names]
    labeled = [n for n, lbl in zip(names, labels) if lbl.exists() and lbl.stat().st_size > 0]
    step = max(1, len(labeled) // max(args.preview, 1))
    preview = set(labeled[::step][: args.preview])
    counts = {"lighting": 0, "blur": 0, "crop_scale": 0}
    dropped = 0
    for i, name in enumerate(names):
        img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  could not read {name}, skipping")
            continue
        h, w = img.shape[:2]
        ids, boxes = read_labels(lbl_dir / (Path(name).stem + ".txt"), w, h)
        for k in range(args.copies):
            aug, keep, new_boxes, applied = augment(
                img, boxes, rng, args.p_light, args.p_blur, args.p_crop
            )
            for a in applied:
                counts[a] += 1
            dropped += len(boxes) - len(keep)
            stem = f"{Path(name).stem}_aug{k}"
            cv2.imwrite(str(out_img / (stem + Path(name).suffix)), aug)
            write_labels(out_lbl / (stem + ".txt"), [ids[j] for j in keep], new_boxes, w, h)
            if name in preview and k == 0:
                pair = np.hstack([draw_boxes(img, boxes), draw_boxes(aug, new_boxes)])
                cv2.imwrite(str(preview_dir / (stem + ".jpg")), pair)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(names)} images")

    data = yaml.safe_load((root / "data.yaml").read_text())
    data["train"] = [f"images/{args.split}", f"images/{out_name}"]
    (root / "data_aug.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    info = {
        "seed": args.seed,
        "split": args.split,
        "copies": args.copies,
        "probabilities": {"lighting": args.p_light, "blur": args.p_blur, "crop": args.p_crop},
        "source_images": len(names),
        "augmented_images": len(names) * args.copies,
        "applied": counts,
        "boxes_dropped_by_crop": dropped,
    }
    (root / "augment_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote {len(names) * args.copies} images to {out_img}")
    print(f"Applied: {counts}; boxes dropped by crop: {dropped}")
    print(f"Train with {root / 'data_aug.yaml'}")
    if args.preview:
        print(f"Before/after previews in {preview_dir}")


if __name__ == "__main__":
    main()
