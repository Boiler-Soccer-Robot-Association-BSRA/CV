"""Take checkerboard photos for calibrate.py from a live camera.

Opens the camera at the resolution you'll calibrate at, shows a live
preview that says whether the whole board is visible, and saves a
lossless photo when you press SPACE. It only saves when the board is
found at full resolution and the board has stopped moving, so every saved
photo is one calibrate.py can use. A 3x3 map shows which parts of the
picture the saved boards have covered: this wide lens needs boards near
every corner and edge (CALIBRATION_GUIDE.md, Step 3).

    python integration-team/calibration/capture.py ~/calibration_photos \\
        --device 0 --width 1920 --height 1200

Keys: SPACE saves, q or Esc quits. When you quit it prints which parts of
the picture still have no board, and the calibrate.py command to run.

Check a folder of photos taken some other way (a webcam app, a phone
script) without a camera:

    python integration-team/calibration/capture.py ~/calibration_photos --check

If the picture is the laptop's own webcam, try --device 1 (or 2).

* **Windows:** if it refuses the size or the preview is very slow, add
  `--backend dshow --fourcc MJPG`. Close any other app using the camera
  (e-CAMView, Teams, the Camera app) first: Windows lets only one app open it.
* **Mac:** the first run asks for camera permission for the terminal app
  (System Settings > Privacy & Security > Camera).
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import IMAGE_SUFFIXES, PATTERN_SIZE, find_corners  # noqa: E402

GRID = 3  # coverage map is GRID x GRID cells of the picture
CELL_NAMES = [
    ["top-left", "top", "top-right"],
    ["left", "center", "right"],
    ["bottom-left", "bottom", "bottom-right"],
]
PREVIEW_WIDTH = 960  # detection in the preview runs at this width, for speed
STILL_PX = 2.0  # board must move less than this (full-res px) between frames to save
NAME_PATTERN = re.compile(r"^capture_(\d+)\.png$")
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 230), (0, 220, 255), (255, 255, 255)


# --- pure helpers (tested without a camera) -----------------------------------------


def next_index(folder: Path) -> int:
    """The number for the next capture_NNN.png, after any already in `folder`."""
    numbers = [
        int(m.group(1))
        for p in folder.iterdir()
        if p.is_file() and (m := NAME_PATTERN.match(p.name))
    ]
    return max(numbers, default=0) + 1


def corner_cells(corners: np.ndarray, size: tuple[int, int], grid: int = GRID) -> set:
    """Which cells of a grid x grid split of the picture the board's corners
    fall in. `corners` is OpenCV's (N, 1, 2) array; `size` is (width, height)."""
    width, height = size
    points = corners.reshape(-1, 2)
    cols = np.clip((points[:, 0] / width * grid).astype(int), 0, grid - 1)
    rows = np.clip((points[:, 1] / height * grid).astype(int), 0, grid - 1)
    return set(zip(rows.tolist(), cols.tolist()))


def missing_cells(covered: set, grid: int = GRID) -> list[str]:
    """Names of the picture's cells no saved board has reached yet."""
    return [CELL_NAMES[r][c] for r in range(grid) for c in range(grid) if (r, c) not in covered]


def board_fraction(corners: np.ndarray, size: tuple[int, int]) -> float:
    """How much of the picture the board's inner corners span (0-1), by the
    area of their outline. Tells near photos from far ones."""
    hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.float32))
    return float(cv2.contourArea(hull)) / (size[0] * size[1])


def board_moved(previous: np.ndarray | None, current: np.ndarray | None) -> float:
    """Largest corner movement in pixels between two detections, or infinity
    if either is missing (no board means "not still")."""
    if previous is None or current is None or previous.shape != current.shape:
        return float("inf")
    return float(np.abs(previous.reshape(-1, 2) - current.reshape(-1, 2)).max())


def detect_preview(gray: np.ndarray, pattern_size: tuple = PATTERN_SIZE) -> np.ndarray | None:
    """Fast, approximate board corners for the live view, in full-res pixels.

    Runs on a frame shrunk to PREVIEW_WIDTH with OpenCV's fast check, so it
    keeps up with the camera; saving re-detects at full resolution.
    """
    scale = min(1.0, PREVIEW_WIDTH / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, pattern_size, flags=flags)
    return corners / scale if found else None


def check_folder(folder: Path, pattern_size: tuple = PATTERN_SIZE) -> dict:
    """What calibrate.py will make of a folder: usable photos, resolution,
    coverage and near/far spread."""
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    usable, unusable, sizes, fractions = [], [], set(), []
    covered: set = set()
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            unusable.append((path.name, "could not read it"))
            continue
        size = (gray.shape[1], gray.shape[0])
        sizes.add(size)
        corners = find_corners(gray, pattern_size)
        if corners is None:
            unusable.append((path.name, "board not found"))
            continue
        usable.append(path.name)
        covered |= corner_cells(corners, size)
        fractions.append(board_fraction(corners, size))
    return {
        "photos": len(paths),
        "usable": usable,
        "unusable": unusable,
        "sizes": sorted(sizes),
        "covered": covered,
        "missing": missing_cells(covered),
        "board_fraction_min": min(fractions, default=0.0),
        "board_fraction_max": max(fractions, default=0.0),
    }


def advice(report: dict) -> list[str]:
    """Plain-language problems with a set of photos, empty if it looks good."""
    problems = []
    if len(report["usable"]) < 20:
        problems.append(f"{len(report['usable'])} usable photos; the guide asks for 20-30")
    if len(report["sizes"]) > 1:
        sizes = ", ".join(f"{w}x{h}" for w, h in report["sizes"])
        problems.append(f"photos have different sizes ({sizes}); calibrate.py will stop")
    if report["missing"]:
        problems.append("no board near: " + ", ".join(report["missing"]))
    if report["usable"] and report["board_fraction_max"] < 0.25:
        problems.append("no close-up photos: bring the board near enough to fill a third")
    if report["usable"] and report["board_fraction_min"] > 0.10:
        problems.append("no far photos: step back until the board is small in the picture")
    return problems


def print_report(report: dict, folder: Path) -> None:
    print(f"{folder}: {len(report['usable'])} of {report['photos']} photos usable")
    for name, reason in report["unusable"]:
        print(f"  not usable: {name} ({reason})")
    if report["sizes"]:
        print("  size: " + ", ".join(f"{w}x{h}" for w, h in report["sizes"]))
    problems = advice(report)
    for problem in problems:
        print(f"  TO DO: {problem}")
    if not problems:
        print("  Looks good: enough photos, every part of the picture, near and far.")
    print(
        f"Next: python integration-team/calibration/calibrate.py {folder} "
        "--output camera_params.yaml --square-mm <measured square size>"
    )


# --- live capture -------------------------------------------------------------------


BACKENDS = {
    "auto": None,
    "dshow": cv2.CAP_DSHOW,  # Windows DirectShow: usually needed for full size on Windows
    "msmf": cv2.CAP_MSMF,  # Windows Media Foundation (OpenCV's Windows default)
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
    "v4l2": cv2.CAP_V4L2,  # Linux
}


def open_camera(
    device: int, width: int, height: int, fourcc: str | None, backend: str = "auto"
) -> cv2.VideoCapture:
    api = BACKENDS[backend]
    capture = cv2.VideoCapture(device) if api is None else cv2.VideoCapture(device, api)
    if not capture.isOpened():
        sys.exit(f"Could not open camera {device}. Is it plugged in, and allowed to be used?")
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def draw_overlay(
    frame: np.ndarray, corners, covered: set, saved: int, message: str, color: tuple
) -> np.ndarray:
    """The preview: the frame shrunk to fit, coverage grid, board, status line."""
    scale = min(1.0, PREVIEW_WIDTH / frame.shape[1])
    view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
    h, w = view.shape[:2]
    overlay = view.copy()
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) in covered:
                top_left = (c * w // GRID, r * h // GRID)
                bottom_right = ((c + 1) * w // GRID, (r + 1) * h // GRID)
                cv2.rectangle(overlay, top_left, bottom_right, GREEN, -1)
    view = cv2.addWeighted(overlay, 0.18, view, 0.82, 0)
    for i in range(1, GRID):
        cv2.line(view, (i * w // GRID, 0), (i * w // GRID, h), WHITE, 1)
        cv2.line(view, (0, i * h // GRID), (w, i * h // GRID), WHITE, 1)
    if corners is not None:
        cv2.drawChessboardCorners(view, PATTERN_SIZE, (corners * scale).astype(np.float32), True)
    status = f"saved {saved} | {message} | SPACE save, q quit"
    cv2.rectangle(view, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(view, status, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return view


def capture_loop(args: argparse.Namespace) -> int:
    args.folder.mkdir(parents=True, exist_ok=True)
    capture = open_camera(args.device, args.width, args.height, args.fourcc, args.backend)
    ok, frame = capture.read()
    if not ok:
        sys.exit(f"Camera {args.device} opened but gave no picture")
    size = (frame.shape[1], frame.shape[0])
    if size != (args.width, args.height) and not args.allow_other_size:
        capture.release()
        sys.exit(
            f"Camera gives {size[0]}x{size[1]}, not the {args.width}x{args.height} asked for. "
            "Calibrate at the size the robot will use: try --fourcc MJPG, check the size the "
            "camera supports, or pass --allow-other-size if this is really the size you want."
        )
    print(f"Camera {args.device}: {size[0]}x{size[1]}. Saving to {args.folder}")

    covered: set = set()
    saved = 0
    previous = None
    message, color = "", WHITE
    message_frames = 0  # how many more frames to show `message` before the live status
    window = "capture.py (SPACE save, q quit)"
    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera stopped giving pictures")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = detect_preview(gray)
        moved = board_moved(previous, corners)
        previous = corners
        if corners is None:
            live, live_color = "no board: show all of it", RED
        elif moved > STILL_PX * 3:
            live, live_color = "board found, hold still", YELLOW
        else:
            live, live_color = "board found", GREEN
        if message_frames > 0:
            message_frames -= 1
        else:
            message = ""
        shown, shown_color = (message, color) if message else (live, live_color)
        cv2.imshow(window, draw_overlay(frame, corners, covered, saved, shown, shown_color))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key != ord(" "):
            continue
        message_frames = 45  # about 2 s at the preview's rate
        if corners is None or moved > STILL_PX:
            message, color = "not saved: hold the whole board still", RED
            continue
        full = find_corners(gray, PATTERN_SIZE)
        if full is None:
            message, color = "not saved: board not found at full size", RED
            continue
        path = args.folder / f"capture_{next_index(args.folder):03d}.png"
        cv2.imwrite(str(path), frame)
        saved += 1
        covered |= corner_cells(full, size)
        message, color = f"saved {path.name}", GREEN
        print(f"  {path.name}  board covers {board_fraction(full, size):.0%} of the picture")

    capture.release()
    cv2.destroyAllWindows()
    print()
    print_report(check_folder(args.folder), args.folder)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path, help="Where to save (or, with --check, check) photos.")
    parser.add_argument("--device", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument(
        "--fourcc",
        default=None,
        help="Pixel format to ask the camera for, e.g. MJPG if full size is slow or refused.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="auto",
        help="Camera API. On Windows, use dshow if the camera won't give full size.",
    )
    parser.add_argument(
        "--allow-other-size",
        action="store_true",
        help="Keep going if the camera won't give --width x --height.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Only check an existing folder of photos."
    )
    args = parser.parse_args()

    if args.check:
        if not args.folder.is_dir():
            parser.error(f"{args.folder} is not a folder")
        print_report(check_folder(args.folder), args.folder)
        return 0
    return capture_loop(args)


if __name__ == "__main__":
    sys.exit(main())
