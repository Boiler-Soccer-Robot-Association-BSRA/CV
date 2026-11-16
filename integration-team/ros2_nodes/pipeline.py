"""The perception pipeline without ROS: frames in, `PerceptionObject`s out.

camera_node.py and perception_node.py are thin ROS 2 wrappers around this
module. Everything that can go wrong in the logic (reading frames, running
the detector, projecting, converting images) lives here, so it can be
tested and run on a machine with no ROS 2 installed (CI, a Mac).

    frame source ──> PerceptionPipeline.process(frame, t) ──> [PerceptionObject, ...]
    (video file,      Detector.predict  (detection-team/inference/detector.py)
     image folder,    project_detections (integration-team/geometry/projection.py)
     or camera)

Run it without ROS, to see positions and check the rate:

    python integration-team/ros2_nodes/pipeline.py \\
        --source path/to/video.mp4 --weights detection-team/models/multiclass_v2.pt \\
        --params camera_params.yaml --frames 300

`--source` is a video file, a folder of images, one image, or a camera
index (0, 1, ...). `--params` is calibrate.py's output plus a `mount:`
section (calibration/CALIBRATION_GUIDE.md, Step 6). The last line reports
the measured rate against the milestone (>= 10 Hz).

Frames follow ARCHITECTURE.md's camera-frame contract: BGR uint8
(height, width, 3).
"""

import argparse
import array
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "integration-team" / "geometry"))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

from projection import BALL_RADIUS_M, CameraParams, project_detections  # noqa: E402

from shared.types import Detection, PerceptionObject  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_HZ = 10.0  # docs/milestones.md: perception messages at >= 10 Hz
DEFAULT_CONF = 0.20  # the operating point in detection-team's results


# --- frame sources ----------------------------------------------------------------


class FrameSource:
    """Frames from a video file, a folder of images, one image, or a camera.

        source = FrameSource("recordings/match.mp4")     # or "frames/", "photo.jpg", 0
        frame, timestamp = source.read()                  # (None, None) when finished
        source.close()

    `timestamp` is the Unix time the frame was read. For a camera that is
    its capture time (to within the driver's buffering). For a file or
    folder it is when the frame was handed out, as if it had just been
    captured, so replayed footage behaves like a live camera downstream.

    `loop=True` restarts a file or folder when it ends (handy for leaving a
    test stream running). `width`/`height` ask a camera for a resolution;
    they are ignored for files. Check `source.frame_size` afterwards: a
    camera may not support the size you asked for.
    """

    def __init__(
        self,
        source: str | int | Path,
        loop: bool = False,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.loop = loop
        self.kind, self.source = classify_source(source)
        self._capture: cv2.VideoCapture | None = None
        self._files: list[Path] = []
        self._index = 0
        if self.kind == "folder":
            folder = Path(self.source)
            self._files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if not self._files:
                raise FileNotFoundError(f"No images found in {folder}")
        elif self.kind == "image":
            self._files = [Path(self.source)]
        else:
            self._capture = cv2.VideoCapture(self.source)
            if not self._capture.isOpened():
                what = "camera" if self.kind == "device" else "video"
                raise OSError(f"Could not open {what} {self.source!r}")
            if self.kind == "device":
                if width:
                    self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """(width, height) for a video or camera; None for image sources,
        whose images may differ in size."""
        if self._capture is None:
            return None
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> tuple[np.ndarray, float] | tuple[None, None]:
        """The next frame and its timestamp, or (None, None) when the source
        has ended (never, for a camera or with loop=True)."""
        frame = self._read_frame()
        if frame is None and self.loop and self.kind != "device":
            self._rewind()
            frame = self._read_frame()
        if frame is None:
            return None, None
        return frame, time.time()

    def _read_frame(self) -> np.ndarray | None:
        if self._capture is not None:
            ok, frame = self._capture.read()
            return frame if ok else None
        if self._index >= len(self._files):
            return None
        path = self._files[self._index]
        self._index += 1
        frame = cv2.imread(str(path))
        if frame is None:
            raise OSError(f"Could not read image: {path}")
        return frame

    def _rewind(self) -> None:
        self._index = 0
        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            frame, timestamp = self.read()
            if frame is None:
                return
            yield frame, timestamp

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()


def classify_source(source: str | int | Path) -> tuple[str, str | int]:
    """What kind of source this is: ("device", 0), ("folder", path),
    ("image", path) or ("video", path).

    A whole number, or a string of digits such as "0" (how a ROS parameter
    or the command line delivers it), is a camera index.
    """
    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        return "device", int(source)
    path = Path(source)
    if path.is_dir():
        return "folder", str(path)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "image", str(path)
    return "video", str(path)


# --- sensor_msgs/Image <-> frame --------------------------------------------------
# Written against the message's fields rather than cv_bridge, so there is no
# compiled ROS dependency (cv_bridge is built against the system numpy, not
# the one in requirements.txt) and it can be tested with a stand-in class.


def frame_to_image_msg(
    frame: np.ndarray, timestamp: float | None, image_cls: Any, frame_id: str = "camera"
) -> Any:
    """A BGR uint8 frame as a `sensor_msgs/msg/Image` (encoding `bgr8`).

    `image_cls` is `sensor_msgs.msg.Image` (or a stand-in in tests). The
    capture time goes in `header.stamp`; None leaves the stamp at zero.
    """
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected a BGR uint8 (H, W, 3) frame, got {frame.dtype} {frame.shape}")
    msg = image_cls()
    msg.height, msg.width = int(frame.shape[0]), int(frame.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    # array('B') is what rclpy stores for uint8[]; handing it over directly
    # skips a per-byte validity check on a multi-megabyte buffer.
    msg.data = array.array("B", np.ascontiguousarray(frame).tobytes())
    msg.header.frame_id = frame_id
    if timestamp is not None:
        msg.header.stamp.sec = int(timestamp)
        msg.header.stamp.nanosec = int(round((timestamp - int(timestamp)) * 1e9)) % 1_000_000_000
    return msg


def image_msg_to_frame(msg: Any) -> tuple[np.ndarray, float | None]:
    """A `sensor_msgs/msg/Image` as (BGR uint8 frame, capture time).

    Accepts `bgr8` and `rgb8` (converted to BGR). A zero stamp means the
    publisher had no capture time, and comes back as None, matching how
    perception_message.py treats an unset timestamp.
    """
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"Unsupported image encoding {msg.encoding!r}; expected bgr8 or rgb8")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    # step can be wider than width * 3 if the driver pads rows.
    frame = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        frame = frame[:, :, ::-1]
    stamp = msg.header.stamp
    timestamp = stamp.sec + stamp.nanosec * 1e-9
    return np.ascontiguousarray(frame), (timestamp if timestamp > 0 else None)


# --- detect + project -------------------------------------------------------------


class DetectorLike(Protocol):
    def predict(self, image: np.ndarray, timestamp: float | None = None) -> list[Detection]: ...


class ResolutionMismatch(ValueError):
    """A frame's size differs from the size the camera was calibrated at, so
    projected positions would be wrong."""


class PerceptionPipeline:
    """Detector + ground-plane projection for one camera.

        pipeline = PerceptionPipeline(Detector("multiclass_v2.pt", conf=0.20), camera_params)
        objects = pipeline.process(frame, timestamp)    # list[PerceptionObject]

    Detections that can't be placed on the floor (at or above the horizon)
    are dropped, as `project_detections` does; `last_detections` keeps the
    raw detections of the latest frame for logging and debugging.

    If the calibration records its resolution and a frame has a different
    size, `process` raises `ResolutionMismatch` instead of returning wrong
    positions. Fix the camera resolution or recalibrate; pass
    `check_resolution=False` only for a dry run where positions don't matter.
    """

    def __init__(
        self,
        detector: DetectorLike,
        camera_params: CameraParams,
        ball_radius_m: float = BALL_RADIUS_M,
        check_resolution: bool = True,
    ) -> None:
        self.detector = detector
        self.camera_params = camera_params
        self.ball_radius_m = ball_radius_m
        self.check_resolution = check_resolution
        self.last_detections: list[Detection] = []

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> list[PerceptionObject]:
        expected = self.camera_params.resolution
        actual = (frame.shape[1], frame.shape[0])
        if self.check_resolution and expected is not None and tuple(expected) != actual:
            raise ResolutionMismatch(
                f"Frame is {actual[0]}x{actual[1]} but the camera was calibrated at "
                f"{expected[0]}x{expected[1]}; projected positions would be wrong"
            )
        self.last_detections = self.detector.predict(frame, timestamp=timestamp)
        return project_detections(self.last_detections, self.camera_params, self.ball_radius_m)


class RateMeter:
    """Counts processed frames and reports how fast they are coming.

    meter.tick()                 # once per processed frame
    hz = meter.take()            # rate since the last take(), then resets
    """

    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self._start = clock()
        self._count = 0
        self.total = 0

    def tick(self) -> None:
        self._count += 1
        self.total += 1

    def take(self) -> float:
        now = self._clock()
        elapsed = now - self._start
        hz = self._count / elapsed if elapsed > 0 else 0.0
        self._start, self._count = now, 0
        return hz


# --- run without ROS --------------------------------------------------------------


def run(
    source: FrameSource,
    pipeline: PerceptionPipeline,
    max_frames: int | None = None,
    on_frame=None,
) -> dict[str, float | int]:
    """Push frames from `source` through `pipeline` as fast as they come.

    Each frame also makes the round trip a ROS image message would
    (frame -> bytes -> frame), so the measured rate includes that copy.
    Returns frames, objects, seconds and hz. `on_frame(index, objects)` is
    called per frame if given.
    """
    frames = objects_total = 0
    start = time.perf_counter()
    for frame, timestamp in source:
        shape = frame.shape
        frame = np.frombuffer(frame.tobytes(), dtype=np.uint8).reshape(shape)
        objects = pipeline.process(frame, timestamp)
        objects_total += len(objects)
        if on_frame is not None:
            on_frame(frames, objects)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            break
    seconds = time.perf_counter() - start
    return {
        "frames": frames,
        "objects": objects_total,
        "seconds": seconds,
        "hz": frames / seconds if seconds > 0 else 0.0,
    }


def format_objects(objects: list[PerceptionObject]) -> str:
    if not objects:
        return "nothing on the floor"
    return ", ".join(
        f"{o.class_name} {o.confidence:.2f} at ({o.position.x:+.2f}, {o.position.y:+.2f}) m"
        for o in objects
    )


def main() -> int:
    from detector import Detector
    from projection import load_camera_params

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="Video, image folder, image or camera #.")
    parser.add_argument("--weights", type=Path, required=True, help=".pt or .onnx weights.")
    parser.add_argument("--params", type=Path, required=True, help="Camera params YAML + mount.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=None, help="Stop after this many frames.")
    parser.add_argument("--loop", action="store_true", help="Restart the file/folder at its end.")
    parser.add_argument("--print-every", type=int, default=30, help="Print every Nth frame.")
    parser.add_argument(
        "--ignore-resolution",
        action="store_true",
        help="Run even if frames aren't the calibrated size (positions will be wrong).",
    )
    args = parser.parse_args()

    source = FrameSource(args.source, loop=args.loop)
    pipeline = PerceptionPipeline(
        Detector(args.weights, conf=args.conf, device=args.device),
        load_camera_params(args.params),
        check_resolution=not args.ignore_resolution,
    )

    def show(index: int, objects: list[PerceptionObject]) -> None:
        if args.print_every and index % args.print_every == 0:
            print(f"frame {index}: {format_objects(objects)}")

    try:
        stats = run(source, pipeline, args.frames, show)
    finally:
        source.close()
    verdict = "meets" if stats["hz"] >= TARGET_HZ else "BELOW"
    print(
        f"{stats['frames']} frames, {stats['objects']} objects in {stats['seconds']:.1f} s: "
        f"{stats['hz']:.1f} Hz ({verdict} the {TARGET_HZ:.0f} Hz target; this machine, "
        f"{args.device}, no ROS transport)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
