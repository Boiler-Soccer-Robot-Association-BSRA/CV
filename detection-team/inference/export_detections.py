"""Write a model's detections for some images to a JSON file.

For integration-team: real `Detection` output (shared/types.py) to build
and test the ROS 2 message conversion and perception node against,
before the camera and the full pipeline exist.

    python detection-team/inference/export_detections.py \\
        detection-team/models/ball_v4.pt path/to/frames/ \\
        --conf 0.20 --out detection-team/samples/ball_v4_detections.json

    # fake capture times, as if the images were frames of a 30 fps stream
    python detection-team/inference/export_detections.py MODEL IMAGES... --fps 30

The JSON is one object: the model, the confidence cutoff, and a list of
frames, each with the image name, its size and its detections. Every
detection has exactly the `Detection` fields: `class_name`, `confidence`,
`bbox` (`x_min`, `y_min`, `x_max`, `y_max` in pixels) and `timestamp`
(seconds, or null when no capture time is known).
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.types import Detection  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def frame_record(name: str, width: int, height: int, detections: list[Detection]) -> dict:
    """One frame as plain JSON-ready data, with box coordinates rounded to 0.1 px."""
    out = []
    for det in detections:
        d = asdict(det)
        d["confidence"] = round(d["confidence"], 3)
        d["bbox"] = {k: round(v, 1) for k, v in d["bbox"].items()}
        out.append(d)
    return {"image": name, "width": width, "height": height, "detections": out}


def collect_images(paths: list[Path]) -> list[Path]:
    images = []
    for p in paths:
        if p.is_dir():
            images += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        else:
            images.append(p)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", help="Model weights, e.g. detection-team/models/ball_v4.pt")
    parser.add_argument("images", nargs="+", type=Path, help="Image files or folders")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence cutoff")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fps", type=float, default=None, help="Give frame i the timestamp i / fps (default: null)"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .json file")
    args = parser.parse_args()

    from detector import Detector  # imported here so --help works without ultralytics

    detector = Detector(args.weights, conf=args.conf, device=args.device)
    frames = []
    for i, path in enumerate(collect_images(args.images)):
        image = cv2.imread(str(path))
        if image is None:
            sys.exit(f"Can't read image: {path}")
        timestamp = i / args.fps if args.fps else None
        detections = detector.predict(image, timestamp=timestamp)
        frames.append(frame_record(path.name, image.shape[1], image.shape[0], detections))

    data = {"model": Path(args.weights).stem, "conf": args.conf, "frames": frames}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    n = sum(len(f["detections"]) for f in frames)
    print(f"Wrote {args.out}: {len(frames)} frames, {n} detections")


if __name__ == "__main__":
    main()
