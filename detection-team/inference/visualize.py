"""Draw bounding boxes and confidence scores on images.

`draw_detections(image, detections)` works on any `list[Detection]`
(shared/types.py) — the dummy detector, stock pretrained YOLO, or our own
trained model once `detector.py` is implemented — so the same picture
can be used to eyeball model output, share failure cases with data-team,
or sanity-check what integration-team is being handed.

Run as a script, it runs YOLO over one or more images and writes an
annotated copy of each, plus a per-image confidence summary:

    python detection-team/inference/visualize.py \\
        detection-team/training/test-images/street_test.jpg

    # a folder of images, only keeping detections at >= 0.5 confidence
    python detection-team/inference/visualize.py path/to/frames/ --conf 0.5

    # a trained checkpoint instead of the stock COCO weights
    python detection-team/inference/visualize.py frame.jpg --weights path/to/best.pt

Like scripts/quickstart.py, the script calls ultralytics directly
because detector.py is still a stub (see ARCHITECTURE.md's "Pipeline
status" table). Once detector.py is implemented, `run_yolo` should call
it instead. With the default stock COCO weights, class names are COCO's
(person, sports ball, ...), not ours from shared/classes.py.
"""

import argparse
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.classes import CLASSES  # noqa: E402
from shared.types import BoundingBox, Detection  # noqa: E402

DEFAULT_WEIGHTS = "yolov8n.pt"  # same stock COCO weights as scripts/quickstart.py
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "visualize"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# BGR. Our own classes get fixed colors so they look the same in every
# picture; anything else (e.g. COCO names) gets a stable color from its name.
PALETTE = [
    (0, 165, 255),  # orange
    (255, 128, 0),  # blue
    (0, 220, 0),  # green
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (0, 0, 255),  # red
    (180, 105, 255),  # pink
]


def class_color(class_name: str) -> tuple[int, int, int]:
    if class_name in CLASSES:
        return PALETTE[CLASSES.index(class_name) % len(PALETTE)]
    return PALETTE[zlib.crc32(class_name.encode()) % len(PALETTE)]


def draw_detections(
    image: np.ndarray, detections: list[Detection], min_confidence: float = 0.0
) -> np.ndarray:
    """Return a copy of a BGR `image` with each detection's box and a
    "class_name 0.87" label drawn on it. Detections below
    `min_confidence` are skipped. The input image is not modified."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    # Scale line width and text with the image so labels stay readable on
    # both a 640px frame and a 2048px photo.
    thickness = max(1, round(min(height, width) / 400))
    font_scale = max(0.4, min(height, width) / 1000)

    for det in sorted(detections, key=lambda d: d.confidence):
        if det.confidence < min_confidence:
            continue
        color = class_color(det.class_name)
        x_min, y_min = int(round(det.bbox.x_min)), int(round(det.bbox.y_min))
        x_max, y_max = int(round(det.bbox.x_max)), int(round(det.bbox.y_max))
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # Put the label above the box, or just inside it if the box touches the top edge.
        label_top = y_min - text_h - baseline
        if label_top < 0:
            label_top = max(y_min, 0)
        label_left = min(max(x_min, 0), max(width - text_w, 0))
        cv2.rectangle(
            canvas,
            (label_left, label_top),
            (label_left + text_w, label_top + text_h + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label,
            (label_left, label_top + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def yolo_results_to_detections(results) -> list[Detection]:
    """Convert ultralytics `Results` into shared `Detection`s, keeping the
    model's own class names."""
    detections = []
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_name=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                )
            )
    return detections


def run_yolo(model, image: np.ndarray, conf: float) -> list[Detection]:
    return yolo_results_to_detections(model.predict(source=image, conf=conf, verbose=False))


def collect_images(paths: list[Path]) -> list[Path]:
    images = []
    for path in paths:
        if path.is_dir():
            images.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            images.append(path)
    return images


def summarize(detections: list[Detection]) -> str:
    if not detections:
        return "  no detections"
    lines = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        b = det.bbox
        lines.append(
            f"  {det.class_name:<14} {det.confidence:.2f}  "
            f"[{b.x_min:.0f}, {b.y_min:.0f}, {b.x_max:.0f}, {b.y_max:.0f}]"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", type=Path, nargs="+", help="Image files and/or folders.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights to run.")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Minimum confidence to keep (default 0.25)."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    image_paths = collect_images(args.images)
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        parser.error(f"Image not found: {missing[0]}")
    if not image_paths:
        parser.error("No images found.")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path}: could not read, skipping")
            continue
        detections = run_yolo(model, image, args.conf)
        out_path = args.out_dir / f"{path.stem}_annotated{path.suffix}"
        cv2.imwrite(str(out_path), draw_detections(image, detections))
        print(f"{path} -> {out_path}\n{summarize(detections)}\n")


if __name__ == "__main__":
    main()
