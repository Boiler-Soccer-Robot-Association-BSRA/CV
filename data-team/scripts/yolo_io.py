"""Small helpers for reading/writing YOLO label files.

YOLO format: one ``.txt`` per image, one line per object:
``<class_id> <cx> <cy> <w> <h>``, all coordinates normalized to 0-1 by the
image's width/height. See the "Dataset schema" section of data-team/README.md.
"""

from pathlib import Path

import numpy as np


def xyxy_to_yolo(box, width: int, height: int) -> tuple:
    """Pixel (x_min, y_min, x_max, y_max) -> normalized (cx, cy, w, h)."""
    x0, y0, x1, y1 = box
    return (
        (x0 + x1) / 2 / width,
        (y0 + y1) / 2 / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    )


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, width: int, height: int) -> tuple:
    """Normalized (cx, cy, w, h) -> pixel (x_min, y_min, x_max, y_max)."""
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def read_labels(path: Path, width: int, height: int):
    """Return (class_ids, boxes) where boxes is an (N, 4) float array in pixels."""
    ids, boxes = [], []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            ids.append(int(parts[0]))
            boxes.append(yolo_to_xyxy(*map(float, parts[1:]), width, height))
    return ids, np.array(boxes, dtype=np.float64).reshape(-1, 4)


def write_labels(path: Path, ids, boxes, width: int, height: int) -> None:
    """Write pixel xyxy boxes as a YOLO label file (empty file = no objects)."""
    lines = []
    for cid, box in zip(ids, boxes):
        cx, cy, w, h = xyxy_to_yolo(box, width, height)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
