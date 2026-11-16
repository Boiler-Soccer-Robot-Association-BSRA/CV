"""Tests for integration-team/ros2_nodes/ without ROS.

CI (and most laptops) have no ROS 2, so:

* pipeline.py, which holds all the logic, is tested directly: frame
  sources, the Image message conversion, detect + project, the rate meter.
* camera_node.py and perception_node.py are run against a small fake
  `rclpy` (parameters, publishers, subscriptions, timers) with the message
  stand-ins from test_perception_message.py. That exercises every line of
  the nodes and their wiring, camera -> image -> detector -> projection ->
  PerceptionObject message. It can't prove the real rclpy accepts the same
  calls; that needs a run inside ROS 2.

The detector is a fake that "finds" objects at known pixels, and the camera
is synthetic with a known pose, so projected positions can be checked.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NODES_DIR = REPO_ROOT / "integration-team" / "ros2_nodes"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(NODES_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import pipeline  # noqa: E402
import projection  # noqa: E402  (pipeline.py puts integration-team/geometry on sys.path)
from test_perception_message import MSG, conversion  # noqa: E402

from shared.types import BoundingBox, Detection, PerceptionObject, Position2D  # noqa: E402

WIDTH, HEIGHT = 640, 400
CAMERA_MATRIX = np.array([[350.0, 0, WIDTH / 2], [0, 350.0, HEIGHT / 2], [0, 0, 1]])
BALL_AT = (2.0, 0.3)
ROBOT_AT = (1.5, -0.4)


def camera_params(resolution=(WIDTH, HEIGHT)) -> "projection.CameraParams":
    return projection.CameraParams(
        camera_matrix=CAMERA_MATRIX,
        distortion=np.zeros(5),
        mount=projection.CameraMount(height_m=0.5, pitch_deg=20.0),
        resolution=resolution,
    )


def box_around(u: float, v: float, half: float = 10.0) -> BoundingBox:
    return BoundingBox(u - half, v - half, u + half, v + half)


class FakeDetector:
    """Returns a ball and a robot standing at known floor positions, plus a
    goalpost whose base is above the horizon (which projection must drop)."""

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[tuple, float | None]] = []

    def predict(self, image, timestamp=None) -> list[Detection]:
        self.calls.append((image.shape, timestamp))
        params = camera_params()
        ball_u, ball_v = projection.ground_to_pixel(*BALL_AT, params, projection.BALL_RADIUS_M)
        robot_u, robot_v = projection.ground_to_pixel(*ROBOT_AT, params)
        robot_box = BoundingBox(robot_u - 20, robot_v - 80, robot_u + 20, robot_v)
        return [
            Detection("ball", 0.9, box_around(ball_u, ball_v), timestamp),
            Detection("robot", 0.8, robot_box, timestamp),
            Detection("goalpost", 0.7, BoundingBox(10, 0, 30, 20), timestamp),
        ]


def numbered_frame(i: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    return np.full((height, width, 3), i, dtype=np.uint8)


def write_images(folder: Path, count: int) -> None:
    for i in range(count):
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), numbered_frame(i, 32, 24))


# --- FrameSource ----------------------------------------------------------------


def test_folder_source_reads_images_in_name_order_then_ends(tmp_path):
    write_images(tmp_path, 3)
    (tmp_path / "notes.txt").write_text("not an image")
    source = pipeline.FrameSource(tmp_path)
    assert source.kind == "folder"
    assert source.frame_size is None
    frames = list(source)
    assert [int(f[0, 0, 0]) for f, _ in frames] == [0, 1, 2]
    assert all(f.dtype == np.uint8 and f.shape == (24, 32, 3) for f, _ in frames)
    assert all(t > 1.7e9 for _, t in frames)  # a Unix time, not a frame counter
    assert source.read() == (None, None)


def test_folder_source_loops_when_asked(tmp_path):
    write_images(tmp_path, 2)
    source = pipeline.FrameSource(tmp_path, loop=True)
    assert [int(source.read()[0][0, 0, 0]) for _ in range(5)] == [0, 1, 0, 1, 0]


def test_single_image_source(tmp_path):
    write_images(tmp_path, 1)
    source = pipeline.FrameSource(tmp_path / "frame_000.png")
    assert source.kind == "image"
    assert len(list(source)) == 1


def test_video_source_reads_every_frame_and_can_loop(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened(), "this OpenCV build can't write MJPG video"
    for i in range(4):
        writer.write(numbered_frame(i * 50, 64, 48))
    writer.release()

    source = pipeline.FrameSource(path)
    assert source.kind == "video"
    assert source.frame_size == (64, 48)
    frames = [f for f, _ in source]
    assert len(frames) == 4
    assert frames[0].shape == (48, 64, 3) and frames[0].dtype == np.uint8
    # MJPG is lossy, so check brightness roughly rather than exactly.
    assert [round(float(f.mean()) / 50) for f in frames] == [0, 1, 2, 3]
    source.close()

    looping = pipeline.FrameSource(path, loop=True)
    assert sum(1 for _ in zip(range(9), looping)) == 9
    looping.close()


def test_source_classification():
    assert pipeline.classify_source(0) == ("device", 0)
    assert pipeline.classify_source("2") == ("device", 2)  # how a ROS parameter delivers it
    with pytest.raises(FileNotFoundError):
        pipeline.classify_source("no/such/video.mp4")


def test_empty_folder_and_unreadable_video_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.FrameSource(tmp_path)
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(OSError):
        pipeline.FrameSource(bad)


# --- sensor_msgs/Image conversion -----------------------------------------------


class FakeImage:
    """Stand-in for sensor_msgs.msg.Image: same fields, same defaults."""

    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), frame_id="")
        self.height = self.width = self.step = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.data = b""


def test_image_message_round_trip_keeps_pixels_and_time():
    frame = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    msg = pipeline.frame_to_image_msg(frame, 1731200000.25, FakeImage, frame_id="head_camera")
    assert (msg.height, msg.width, msg.step, msg.encoding) == (HEIGHT, WIDTH, WIDTH * 3, "bgr8")
    assert msg.header.frame_id == "head_camera"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (1731200000, 250000000)
    assert len(msg.data) == HEIGHT * WIDTH * 3
    back, timestamp = pipeline.image_msg_to_frame(msg)
    assert np.array_equal(back, frame)
    assert timestamp == pytest.approx(1731200000.25)


def test_image_message_without_a_time_reads_back_as_none():
    msg = pipeline.frame_to_image_msg(numbered_frame(7), None, FakeImage)
    assert pipeline.image_msg_to_frame(msg)[1] is None


def test_rgb_and_padded_rows_are_handled_and_other_encodings_rejected():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 0] = 200  # blue, in BGR
    msg = FakeImage()
    msg.height, msg.width, msg.step, msg.encoding = 2, 2, 8, "rgb8"  # 2 padding bytes per row
    rgb_rows = np.concatenate([frame[:, :, ::-1].reshape(2, 6), np.zeros((2, 2), np.uint8)], axis=1)
    msg.data = rgb_rows.tobytes()
    back, _ = pipeline.image_msg_to_frame(msg)
    assert np.array_equal(back, frame)
    msg.encoding = "yuv422"
    with pytest.raises(ValueError, match="yuv422"):
        pipeline.image_msg_to_frame(msg)


def test_frame_to_image_msg_rejects_frames_that_break_the_contract():
    with pytest.raises(ValueError):
        pipeline.frame_to_image_msg(np.zeros((4, 4), dtype=np.uint8), None, FakeImage)
    with pytest.raises(ValueError):
        pipeline.frame_to_image_msg(np.zeros((4, 4, 3), dtype=np.float32), None, FakeImage)


# --- PerceptionPipeline ---------------------------------------------------------


def test_pipeline_projects_detections_and_drops_ones_above_the_horizon():
    detector = FakeDetector()
    perception = pipeline.PerceptionPipeline(detector, camera_params())
    objects = perception.process(numbered_frame(0), timestamp=123.5)
    assert [o.class_name for o in objects] == ["ball", "robot"]  # goalpost dropped
    ball, robot = objects
    assert (ball.position.x, ball.position.y) == pytest.approx(BALL_AT, abs=1e-6)
    assert (robot.position.x, robot.position.y) == pytest.approx(ROBOT_AT, abs=1e-6)
    assert ball.confidence == 0.9 and ball.timestamp == 123.5
    assert detector.calls == [((HEIGHT, WIDTH, 3), 123.5)]
    assert len(perception.last_detections) == 3


def test_pipeline_uses_the_given_ball_radius():
    default = pipeline.PerceptionPipeline(FakeDetector(), camera_params())
    alternate = pipeline.PerceptionPipeline(
        FakeDetector(), camera_params(), ball_radius_m=projection.BALL_RADIUS_ALT_M
    )
    near = default.process(numbered_frame(0))[0].position.x
    assert alternate.process(numbered_frame(0))[0].position.x < near - 0.01


def test_pipeline_refuses_frames_at_the_wrong_resolution():
    perception = pipeline.PerceptionPipeline(FakeDetector(), camera_params((1920, 1200)))
    with pytest.raises(pipeline.ResolutionMismatch, match="1920x1200"):
        perception.process(numbered_frame(0))
    lenient = pipeline.PerceptionPipeline(
        FakeDetector(), camera_params((1920, 1200)), check_resolution=False
    )
    assert lenient.process(numbered_frame(0))
    unknown = pipeline.PerceptionPipeline(FakeDetector(), camera_params(None))
    assert unknown.process(numbered_frame(0))


def test_rate_meter_reports_rate_since_last_take():
    now = [0.0]
    meter = pipeline.RateMeter(clock=lambda: now[0])
    for _ in range(30):
        meter.tick()
    now[0] = 2.0
    assert meter.take() == pytest.approx(15.0)
    now[0] = 4.0
    assert meter.take() == 0.0
    assert meter.total == 30


def test_run_pushes_frames_through_and_stops_at_max_frames(tmp_path):
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"frame_{i}.png"), numbered_frame(i))
    seen = []
    stats = pipeline.run(
        pipeline.FrameSource(tmp_path),
        pipeline.PerceptionPipeline(FakeDetector(), camera_params()),
        max_frames=3,
        on_frame=lambda index, objects: seen.append((index, len(objects))),
    )
    assert stats["frames"] == 3 and stats["objects"] == 6
    assert stats["hz"] > 0
    assert seen == [(0, 2), (1, 2), (2, 2)]
    assert "ball 0.90 at (+2.00, +0.30) m" in pipeline.format_objects(
        [PerceptionObject("ball", 0.9, Position2D(2.0, 0.3))]
    )


# --- the nodes, against a fake rclpy --------------------------------------------


class FakeTopic:
    def __init__(self) -> None:
        self.callbacks = []
        self.messages = []


class FakeRos:
    """The handful of rclpy features the nodes use, kept in one place so
    two nodes can talk over a topic inside a test."""

    def __init__(self) -> None:
        self.topics: dict[str, FakeTopic] = {}
        self.parameters: dict[str, object] = {}
        self.logs: list[tuple[str, str]] = []
        ros = self

        class Timer:
            def __init__(self, period, callback) -> None:
                self.period, self.callback, self.cancelled = period, callback, False

            def cancel(self) -> None:
                self.cancelled = True

        class Publisher:
            def __init__(self, msg_type, topic) -> None:
                self.msg_type, self.topic = msg_type, ros.topics.setdefault(topic, FakeTopic())

            def publish(self, msg) -> None:
                assert isinstance(msg, self.msg_type), f"published a {type(msg).__name__}"
                self.topic.messages.append(msg)
                for callback in self.topic.callbacks:
                    callback(msg)

        class Node:
            def __init__(self, name: str) -> None:
                self.name, self._parameters, self.timers = name, {}, []
                self.destroyed = False

            def declare_parameter(self, name, default):
                self._parameters[name] = ros.parameters.get(name, default)

            def get_parameter(self, name):
                return SimpleNamespace(value=self._parameters[name])  # KeyError if undeclared

            def create_publisher(self, msg_type, topic, qos):
                return Publisher(msg_type, topic)

            def create_subscription(self, msg_type, topic, callback, qos):
                ros.topics.setdefault(topic, FakeTopic()).callbacks.append(callback)
                return callback

            def create_timer(self, period, callback):
                self.timers.append(Timer(period, callback))
                return self.timers[-1]

            def get_logger(self):
                def log(level):
                    return lambda text: ros.logs.append((level, text))

                return SimpleNamespace(info=log("info"), warn=log("warn"), error=log("error"))

            def destroy_node(self) -> None:
                self.destroyed = True

        self.Node = Node

    def install(self, monkeypatch) -> None:
        def module(name: str, **attrs) -> ModuleType:
            mod = ModuleType(name)
            for key, value in attrs.items():
                setattr(mod, key, value)
            monkeypatch.setitem(sys.modules, name, mod)
            return mod

        module("rclpy", init=lambda args=None: None, ok=lambda: True, shutdown=lambda: None)
        module("rclpy.node", Node=self.Node)
        module("sensor_msgs")
        module("sensor_msgs.msg", Image=FakeImage)
        module("my_robot_controller", msg=MSG)
        module("my_robot_controller.msg", **vars(MSG))


def load_node(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", NODES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ros(monkeypatch) -> FakeRos:
    fake = FakeRos()
    fake.install(monkeypatch)
    return fake


def test_camera_node_publishes_each_frame_then_reports_the_end(ros, tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"frame_{i}.png"), numbered_frame(i))
    ros.parameters.update(source=str(tmp_path), fps=20.0, frame_id="head_camera")
    node = load_node("camera_node").CameraNode()
    publish_timer = node.timers[0]
    assert publish_timer.period == pytest.approx(0.05)

    for _ in range(4):
        publish_timer.callback()
    images = ros.topics["camera/image_raw"].messages
    assert len(images) == 3
    assert all(m.encoding == "bgr8" and (m.width, m.height) == (WIDTH, HEIGHT) for m in images)
    assert images[0].header.frame_id == "head_camera"
    assert images[0].header.stamp.sec > 1_700_000_000
    assert [int(pipeline.image_msg_to_frame(m)[0][0, 0, 0]) for m in images] == [0, 1, 2]
    assert node.done and publish_timer.cancelled
    assert any("Source ended after 3 frames" in text for _, text in ros.logs)
    node.timers[1].callback()  # the rate log must not crash
    node.destroy_node()
    assert node.destroyed


def test_camera_node_needs_a_source(ros):
    with pytest.raises(ValueError, match="source"):
        load_node("camera_node").CameraNode()


def test_camera_to_perception_end_to_end(ros, tmp_path, monkeypatch):
    """Frames from a folder -> camera node -> image topic -> perception node
    -> PerceptionObject messages at the right floor positions."""
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"frame_{i}.png"), numbered_frame(i))
    params_file = tmp_path / "camera_params.yaml"
    params_file.write_text(
        f"camera_matrix: {CAMERA_MATRIX.tolist()}\n"
        "distortion: [0, 0, 0, 0, 0]\n"
        f"resolution: {{width: {WIDTH}, height: {HEIGHT}}}\n"
        "mount: {height_m: 0.5, pitch_deg: 20.0}\n"
    )
    ros.parameters.update(
        source=str(tmp_path), weights="multiclass_v2.pt", camera_params=str(params_file)
    )
    perception_module = load_node("perception_node")
    monkeypatch.setattr(perception_module, "Detector", FakeDetector)
    perception = perception_module.PerceptionNode()
    camera = load_node("camera_node").CameraNode()

    for _ in range(2):
        camera.timers[0].callback()

    published = ros.topics["perception/objects"].messages
    assert [m.class_name for m in published] == ["ball", "robot", "ball", "robot"]
    assert all(isinstance(m, MSG.PerceptionObject) for m in published)
    ball = conversion.from_ros_message(published[0])
    assert (ball.position.x, ball.position.y) == pytest.approx(BALL_AT, abs=1e-4)
    # Every object of one frame carries that frame's capture time.
    image_times = [
        pipeline.image_msg_to_frame(m)[1] for m in ros.topics["camera/image_raw"].messages
    ]
    assert [m.timestamp for m in published] == pytest.approx(
        [image_times[0], image_times[0], image_times[1], image_times[1]]
    )
    assert perception.objects_published == 4 and perception.meter.total == 2
    perception.timers[0].callback()  # the rate log must not crash
    assert any("objects published" in text for _, text in ros.logs)


def test_perception_node_needs_weights_and_camera_params(ros):
    with pytest.raises(ValueError, match="weights"):
        load_node("perception_node").PerceptionNode()
