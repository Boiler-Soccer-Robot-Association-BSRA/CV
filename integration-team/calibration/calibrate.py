"""Camera intrinsic calibration and lens distortion correction.

Finds a printed checkerboard in a folder of photos and saves the camera
matrix and distortion coefficients for integration-team/geometry/projection.py.
See CALIBRATION_GUIDE.md for how to print the board and take the photos.

    python integration-team/calibration/calibrate.py ~/calibration_photos \\
        --output camera_params.yaml --square-mm 35.0
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PATTERN_SIZE = (9, 6)  # inner corners of the 10x7-square board
MIN_PHOTOS = 3  # OpenCV's minimum for a usable result; the guide asks for more
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points(pattern_size: tuple, square_mm: float) -> np.ndarray:
    """The board's inner corners in mm, on its own flat (z = 0) plane."""
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return points


def find_corners(gray: np.ndarray, pattern_size: tuple) -> np.ndarray | None:
    """Sub-pixel board corners in a grayscale photo, or None if not found."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)


def calibrate(images_dir: Path, square_mm: float, pattern_size: tuple = PATTERN_SIZE) -> dict:
    """Calibrate from every usable photo in `images_dir`; returns the params."""
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        sys.exit(f"No images found in {images_dir}")

    image_points = []
    resolution = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skipped {path.name}: could not read it")
            continue
        size = (gray.shape[1], gray.shape[0])
        if resolution not in (None, size):
            sys.exit(f"{path.name} is {size}, but earlier photos are {resolution}")
        corners = find_corners(gray, pattern_size)
        if corners is None:
            print(f"  skipped {path.name}: board not found")
            continue
        resolution = size
        image_points.append(corners)

    used = len(image_points)
    print(f"Used {used} of {len(paths)} photos")
    if used < MIN_PHOTOS:
        sys.exit(f"Need at least {MIN_PHOTOS} photos with the whole board visible")

    object_points = [board_points(pattern_size, square_mm)] * used
    error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, resolution, None, None
    )
    return {
        "resolution": {"width": resolution[0], "height": resolution[1]},
        "camera_matrix": camera_matrix.tolist(),
        "distortion": distortion.ravel().tolist(),
        "reprojection_error_px": float(error),
        "photos_used": used,
        "pattern_size": list(pattern_size),
        "square_mm": square_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "images_dir", type=Path, help="Directory of checkerboard calibration images."
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to save camera params.")
    parser.add_argument(
        "--square-mm",
        type=float,
        default=35.0,
        help="Measured square size of the printed board, in mm (default 35.0).",
    )
    args = parser.parse_args()

    params = calibrate(args.images_dir, args.square_mm)
    args.output.write_text(yaml.safe_dump(params, sort_keys=False))
    print(f"Reprojection error: {params['reprojection_error_px']:.3f} px (aim for under 0.5)")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
