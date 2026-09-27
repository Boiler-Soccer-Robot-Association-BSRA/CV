"""Generate the tiny synthetic dataset in fixtures/sample_dataset/.

Not run automatically — its output (a handful of small PNGs plus
manifest.yaml) is committed so the benchmark harness and its tests have
a fixed, deterministic dataset to run against without downloading
anything. Re-run this only if you intend to change the fixture, then
commit the new output alongside it.

Each frame is a dark green "field" with a bright circular "ball" at a
known position, so the ground-truth bounding box is exactly the
circle's bounding square. One frame has no ball at all, to exercise the
no-detections case. This is deliberately not photorealistic — it's a
fixture for exercising the harness's IoU-matching and metrics code, not
a stand-in for a real labeled dataset.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

OUT_DIR = Path(__file__).parent / "sample_dataset"
IMAGE_SIZE = 64
FIELD_COLOR_BGR = (20, 80, 20)
BALL_COLOR_BGR = (30, 220, 245)

# (center_x, center_y, radius) in pixels, or None for a frame with no ball.
FRAMES = [
    (32, 32, 14),
    (18, 20, 10),
    (48, 45, 12),
    (10, 50, 8),
    None,
]


def make_frame(ball: tuple[int, int, int] | None) -> tuple[np.ndarray, list[dict]]:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), FIELD_COLOR_BGR, dtype=np.uint8)
    if ball is None:
        return image, []

    cx, cy, r = ball
    cv2.circle(image, (cx, cy), r, BALL_COLOR_BGR, thickness=-1, lineType=cv2.LINE_AA)
    bbox = [
        float(max(cx - r, 0)),
        float(max(cy - r, 0)),
        float(min(cx + r, IMAGE_SIZE)),
        float(min(cy + r, IMAGE_SIZE)),
    ]
    return image, [{"class_name": "ball", "bbox": bbox}]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, ball in enumerate(FRAMES):
        image, annotations = make_frame(ball)
        filename = f"frame_{i:03d}.png"
        cv2.imwrite(str(OUT_DIR / filename), image)
        manifest.append({"image": filename, "annotations": annotations})

    manifest_path = OUT_DIR / "manifest.yaml"
    with manifest_path.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Wrote {len(manifest)} frames and {manifest_path}")


if __name__ == "__main__":
    main()
