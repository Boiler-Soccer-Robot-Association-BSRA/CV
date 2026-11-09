"""Try ground-plane projection by eye, on a webcam, video or photo.

Draws a floor grid (lines every 0.5 m ahead, plus left/right lines) using
projection.py's math, labels each detection with its projected position,
and prints the floor position of any pixel you click. Put tape on the
floor at measured distances and check the drawn lines land on it.

    # Laptop webcam, no calibration yet: approximate the lens from its field of view.
    python integration-team/geometry/projection_demo.py --source 0 \\
        --height 0.75 --pitch 30 --hfov-deg 70

    # A calibrated camera (calibrate.py output, with or without a mount: section).
    python integration-team/geometry/projection_demo.py --source 0 \\
        --params camera_params.yaml --height 0.45 --pitch 20

    # A photo, saved to a file instead of opening a window.
    python integration-team/geometry/projection_demo.py --source photo.jpg \\
        --height 0.75 --pitch 30 --save out.png

Keys: q / Esc quits. Click anywhere to print that pixel's floor position.

Detections use detection-team's Detector. The default stock `yolov8n.pt`
only knows COCO's "sports ball" (shown as `ball`); pass `--weights` with a
team model (e.g. ball_v4.pt from the team drive) for real results, or
`--no-detect` to show only the grid.

Without --params the lens is treated as distortion-free with the given
horizontal field of view, so expect errors of several percent, more toward
the image edges. That's fine for a sanity check, not for accuracy numbers.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

from projection import (  # noqa: E402
    BALL_RADIUS_M,
    CameraMount,
    CameraParams,
    ProjectionError,
    contact_pixel,
    ground_to_pixel,
    load_camera_params,
    pixel_to_ground,
)

GRID_AHEAD_M = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
GRID_SIDE_M = [-1.0, -0.5, 0.0, 0.5, 1.0]
GRID_COLOR = (0, 200, 255)  # BGR orange
BOX_COLOR = (0, 255, 0)


def approx_camera_matrix(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole intrinsics for a camera with this horizontal field of view,
    square pixels and the principal point at the image center."""
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def build_params(args, width: int, height: int) -> CameraParams:
    """CameraParams from --params (if given) or --hfov-deg, with the mount
    from the command line overriding any mount in the file."""
    mount = None
    if args.height is not None:
        mount = CameraMount(
            height_m=args.height, pitch_deg=args.pitch, roll_deg=args.roll, yaw_deg=args.yaw
        )
    if args.params:
        params = load_camera_params(args.params, mount=mount)
        if params.resolution and params.resolution != (width, height):
            print(
                f"WARNING: {args.params} was calibrated at {params.resolution}, "
                f"but frames are {(width, height)}. Positions will be wrong."
            )
        return params
    if mount is None:
        sys.exit("Without --params, give the mount: --height (and usually --pitch)")
    return CameraParams(
        camera_matrix=approx_camera_matrix(width, height, args.hfov_deg),
        distortion=np.zeros(5),
        mount=mount,
        resolution=(width, height),
    )


def _to_int(point) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _text(frame: np.ndarray, text: str, origin, color) -> None:
    """Text sized to the frame, so labels stay readable on large photos."""
    scale = max(0.5, frame.shape[1] / 1280 * 0.6)
    thickness = max(1, round(scale * 2))
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _visible_pixel(x: float, y: float, params: CameraParams, size) -> tuple[int, int] | None:
    """The pixel for floor point (x, y), or None if it's off-screen/behind."""
    try:
        u, v = ground_to_pixel(x, y, params)
    except ProjectionError:
        return None
    width, height = size
    if -width <= u <= 2 * width and -height <= v <= 2 * height:  # allow clipping by cv2.line
        return _to_int((u, v))
    return None


def draw_grid(frame: np.ndarray, params: CameraParams) -> None:
    """Floor lines at fixed distances ahead and to the sides, as the
    projection math thinks they appear."""
    size = (frame.shape[1], frame.shape[0])
    side_min, side_max = min(GRID_SIDE_M), max(GRID_SIDE_M)
    for ahead in GRID_AHEAD_M:
        a = _visible_pixel(ahead, side_max, params, size)
        b = _visible_pixel(ahead, side_min, params, size)
        if a and b:
            cv2.line(frame, a, b, GRID_COLOR, 1, cv2.LINE_AA)
            label = _visible_pixel(ahead, 0.0, params, size)
            _text(frame, f"{ahead:g} m", label, GRID_COLOR)
    for side in GRID_SIDE_M:
        points = [_visible_pixel(ahead, side, params, size) for ahead in GRID_AHEAD_M]
        points = [p for p in points if p]
        for a, b in zip(points, points[1:], strict=False):
            cv2.line(frame, a, b, GRID_COLOR, 1, cv2.LINE_AA)


def draw_detections(frame, detections, params: CameraParams, ball_radius_m: float) -> None:
    """Each box, its projected position, and the pixel that was projected."""
    for det in detections:
        box = det.bbox
        cv2.rectangle(
            frame, _to_int((box.x_min, box.y_min)), _to_int((box.x_max, box.y_max)), BOX_COLOR, 2
        )
        u, v, plane = contact_pixel(det, ball_radius_m)
        cv2.circle(frame, _to_int((u, v)), 4, (0, 0, 255), -1)
        try:
            pos = pixel_to_ground(u, v, params, plane)
            text = (
                f"{det.class_name} x={pos.x:.2f} y={pos.y:+.2f} ({math.hypot(pos.x, pos.y):.2f} m)"
            )
        except ProjectionError:
            text = f"{det.class_name}: above horizon"
        _text(frame, text, _to_int((box.x_min, max(15, box.y_min - 6))), BOX_COLOR)


def make_detector(args):
    """detection-team's Detector, or None with --no-detect."""
    if args.no_detect:
        return None
    from detector import COCO_CLASS_MAP, Detector

    stock = Path(args.weights).name.startswith("yolov8")
    return Detector(args.weights, conf=args.conf, class_map=COCO_CLASS_MAP if stock else None)


def annotate(frame, params, detector, ball_radius_m) -> np.ndarray:
    out = frame.copy()
    draw_grid(out, params)
    if detector is not None:
        draw_detections(out, detector.predict(frame), params, ball_radius_m)
    m = params.mount
    _text(
        out,
        f"height {m.height_m:.2f} m  pitch {m.pitch_deg:.1f} deg  (q to quit, click = position)",
        (10, out.shape[0] - 10),
        (255, 255, 255),
    )
    return out


def open_source(source: str):
    """(capture, still_image): a camera/video capture, or a loaded image."""
    if source.isdigit():
        return cv2.VideoCapture(int(source)), None
    image = cv2.imread(source)
    if image is not None:
        return None, image
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        sys.exit(f"Could not open {source} as a camera, image or video")
    return capture, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default="0", help="Camera index, image or video path")
    parser.add_argument("--params", type=Path, help="calibrate.py YAML (optional)")
    parser.add_argument("--hfov-deg", type=float, default=70.0, help="Used without --params")
    parser.add_argument("--height", type=float, help="Lens height above the floor, m")
    parser.add_argument("--pitch", type=float, default=0.0, help="Downward tilt, degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll, degrees")
    parser.add_argument("--yaw", type=float, default=0.0, help="Left turn, degrees")
    parser.add_argument("--weights", default="yolov8n.pt", help="Detector weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ball-radius", type=float, default=BALL_RADIUS_M, help="m")
    parser.add_argument("--no-detect", action="store_true", help="Grid and clicks only")
    parser.add_argument("--save", type=Path, help="Write one annotated frame here and exit")
    args = parser.parse_args()

    capture, image = open_source(args.source)
    frame = image if image is not None else capture.read()[1]
    if frame is None:
        sys.exit(f"No frame from {args.source}")
    params = build_params(args, frame.shape[1], frame.shape[0])
    detector = make_detector(args)

    if args.save:
        cv2.imwrite(str(args.save), annotate(frame, params, detector, args.ball_radius))
        print(f"Saved {args.save}")
        return

    def on_click(event, u, v, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        try:
            pos = pixel_to_ground(u, v, params)
            print(f"pixel ({u}, {v}) -> floor x={pos.x:.3f} m, y={pos.y:+.3f} m")
        except ProjectionError:
            print(f"pixel ({u}, {v}) is at or above the horizon")

    window = "projection demo"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        cv2.imshow(window, annotate(frame, params, detector, args.ball_radius))
        if cv2.waitKey(1 if capture else 30) & 0xFF in (ord("q"), 27):
            break
        if capture is not None:
            ok, next_frame = capture.read()
            if not ok:
                break
            frame = next_frame
    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
