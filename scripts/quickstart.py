"""Run one sample image through as much of the real BSRA-CV pipeline as
currently exists.

WHAT THIS RUNS TODAY:
  - Loads an image (yours, or ultralytics' bundled sample image).
  - Runs it through detection-team's `Detector`
    (detection-team/inference/detector.py) with stock pretrained YOLO
    weights, or a trained checkpoint via --weights.
  - Prints the resulting shared `Detection`s (shared/types.py).

WHAT THIS DOES NOT COVER (because the code doesn't exist yet):
  - A RoboCup-trained model, by default: without --weights this uses
    YOLO's stock pretrained COCO weights, which do NOT know about soccer
    balls, goalposts, or field lines the way a fine-tuned model does.
    Detections you see are whatever COCO classes (person, sports ball,
    etc.) happen to appear in the image.
  - Camera calibration / lens distortion correction
    (integration-team/calibration/).
  - Ground-plane projection to a robot-relative position
    (integration-team/geometry/).
  - The ROS 2 perception message / nodes
    (integration-team/messages/, integration-team/ros2_nodes/).

As each of those gets built out for real, extend this script to call
into it instead of skipping it, so it stays the one command that proves
the current end-to-end state of the pipeline.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

import cv2  # noqa: E402
from detector import Detector  # noqa: E402

from shared.types import Detection  # noqa: E402

DEFAULT_WEIGHTS = "yolov8n.pt"  # pretrained COCO weights, auto-downloaded on first run


def run(image_path: Path, weights: str = DEFAULT_WEIGHTS) -> list[Detection]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")
    return Detector(weights).predict(image)


def default_sample_image() -> Path:
    try:
        from ultralytics.utils import ASSETS

        return ASSETS / "bus.jpg"
    except (ImportError, AttributeError) as e:
        raise SystemExit(
            "No --image given and couldn't locate ultralytics' bundled sample "
            f"image ({e}). Pass one explicitly: "
            "python scripts/quickstart.py path/to/image.jpg"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a sample image. If omitted, uses ultralytics' bundled sample image.",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="YOLO weights to run (default: stock pretrained COCO yolov8n.pt).",
    )
    args = parser.parse_args()

    image_path = args.image or default_sample_image()
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    if args.weights == DEFAULT_WEIGHTS:
        print(f"Running pretrained (stock COCO, not RoboCup-trained) YOLO on: {image_path}\n")
    else:
        print(f"Running {args.weights} on: {image_path}\n")
    detections = run(image_path, args.weights)

    if not detections:
        print("No detections.")
    for d in detections:
        print(d)


if __name__ == "__main__":
    main()
