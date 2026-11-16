"""ROS 2 node: camera frames in, robot-relative `PerceptionObject` messages out.

Subscribes to camera images, runs detection-team's `Detector` on each
frame, projects every detection onto the floor with integration-team's
`projection.py`, and publishes one `my_robot_controller/msg/PerceptionObject`
per object (schema in ARCHITECTURE.md's "Data contracts").

    # after `colcon build` in integration-team/ros_ws and `source install/setup.bash`
    python3 integration-team/ros2_nodes/perception_node.py --ros-args \\
        -p weights:=detection-team/models/multiclass_v2.pt \\
        -p camera_params:=camera_params.yaml

Parameters:

| Name            | Default              | Meaning                                         |
|-----------------|----------------------|-------------------------------------------------|
| `weights`       | (required)           | `.pt` or `.onnx` weights (multiclass_v2)        |
| `camera_params` | (required)           | calibrate.py's YAML plus a `mount:` section     |
| `conf`          | 0.20                 | Minimum detection confidence                    |
| `device`        | `cpu`                | `cpu`, `cuda:0`, ...                            |
| `ball_radius_m` | 0.05                 | 0.05 SPL ball; 0.07 for the FIFA mini ball      |
| `image_topic`   | `camera/image_raw`   | Where camera_node (or a camera driver) publishes|
| `output_topic`  | `perception/objects` | Where the objects are published                 |

**One message per object.** The schema has no list message, so a frame
with three objects is three messages with the same `timestamp` (the
frame's capture time), and a frame with nothing on the floor publishes
nothing. Subscribers that need "everything seen in one frame" group by
`timestamp`.

**Rate.** Frames are handled one at a time; the subscription keeps only
the newest frame, so if detection is slower than the camera the node skips
frames instead of falling behind. It logs frames per second every 5 s and
warns when that drops below the 10 Hz target.

The detect-and-project logic is `PerceptionPipeline` in pipeline.py, which
has no ROS in it and is what the tests cover. This file needs a sourced
ROS 2 environment with the workspace built (rclpy, sensor_msgs,
my_robot_controller.msg), and the root requirements.txt installed in the
Python that ROS uses.
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "integration-team" / "messages"))
sys.path.insert(0, str(REPO_ROOT / "integration-team" / "geometry"))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

from detector import Detector  # noqa: E402
from my_robot_controller import msg as perception_msgs  # noqa: E402
from perception_message import to_ros_message  # noqa: E402
from pipeline import (  # noqa: E402
    DEFAULT_CONF,
    TARGET_HZ,
    PerceptionPipeline,
    RateMeter,
    image_msg_to_frame,
)
from projection import BALL_RADIUS_M, load_camera_params  # noqa: E402

LOG_EVERY_S = 5.0


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("weights", "")
        self.declare_parameter("camera_params", "")
        self.declare_parameter("conf", DEFAULT_CONF)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("ball_radius_m", BALL_RADIUS_M)
        self.declare_parameter("image_topic", "camera/image_raw")
        self.declare_parameter("output_topic", "perception/objects")

        weights = str(self.get_parameter("weights").value)
        camera_params = str(self.get_parameter("camera_params").value)
        if not weights or not camera_params:
            raise ValueError("perception_node needs the 'weights' and 'camera_params' parameters")
        image_topic = str(self.get_parameter("image_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)

        detector = Detector(
            weights,
            conf=float(self.get_parameter("conf").value),
            device=str(self.get_parameter("device").value),
        )
        self.pipeline = PerceptionPipeline(
            detector,
            load_camera_params(camera_params),
            ball_radius_m=float(self.get_parameter("ball_radius_m").value),
        )
        self.publisher = self.create_publisher(perception_msgs.PerceptionObject, output_topic, 10)
        # Depth 1: always work on the newest frame rather than a backlog.
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 1)
        self.meter = RateMeter()
        self.objects_published = 0
        self.log_timer = self.create_timer(LOG_EVERY_S, self.log_rate)
        self.get_logger().info(
            f"Detecting with {weights} on '{image_topic}', publishing on '{output_topic}'"
        )

    def on_image(self, image_msg) -> None:
        frame, timestamp = image_msg_to_frame(image_msg)
        for obj in self.pipeline.process(frame, timestamp):
            self.publisher.publish(to_ros_message(obj, perception_msgs))
            self.objects_published += 1
        self.meter.tick()

    def log_rate(self) -> None:
        hz = self.meter.take()
        message = f"{hz:.1f} frames/s, {self.objects_published} objects published so far"
        if 0 < hz < TARGET_HZ:
            self.get_logger().warn(f"{message} (below the {TARGET_HZ:.0f} Hz target)")
        else:
            self.get_logger().info(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
