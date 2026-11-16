"""ROS 2 node that publishes camera frames as `sensor_msgs/msg/Image`.

    # replay a video file or a folder of images (no camera needed)
    python3 integration-team/ros2_nodes/camera_node.py --ros-args \\
        -p source:=/path/to/video.mp4 -p fps:=30.0 -p loop:=true

    # a real camera, by device index
    python3 integration-team/ros2_nodes/camera_node.py --ros-args \\
        -p source:="'0'" -p width:=1920 -p height:=1200

Parameters:

| Name       | Default            | Meaning                                              |
|------------|--------------------|------------------------------------------------------|
| `source`   | (required)         | Video file, image folder, one image, or camera index |
| `fps`      | 30.0               | Publish rate                                         |
| `loop`     | false              | Restart a file/folder when it ends                   |
| `width`, `height` | 0 (driver default) | Resolution to ask a camera for                |
| `topic`    | `camera/image_raw` | Topic to publish on                                  |
| `frame_id` | `camera`           | `header.frame_id` of each image                      |

Each image is `bgr8` (ARCHITECTURE.md's camera-frame contract) with the
frame's capture time in `header.stamp`. When a file or folder ends and
`loop` is false, the node logs it and exits.

The frame reading itself is `FrameSource` in pipeline.py, which has no ROS
in it and is what the tests cover. This file needs a sourced ROS 2
environment (rclpy, sensor_msgs).
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import FrameSource, RateMeter, frame_to_image_msg  # noqa: E402

LOG_EVERY_S = 5.0


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")
        self.declare_parameter("source", "")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("width", 0)
        self.declare_parameter("height", 0)
        self.declare_parameter("topic", "camera/image_raw")
        self.declare_parameter("frame_id", "camera")

        source = str(self.get_parameter("source").value)
        if not source:
            raise ValueError("camera_node needs the 'source' parameter (file, folder or index)")
        fps = float(self.get_parameter("fps").value)
        if fps <= 0:
            raise ValueError(f"fps must be above 0, got {fps}")
        self.frame_id = str(self.get_parameter("frame_id").value)
        topic = str(self.get_parameter("topic").value)

        self.source = FrameSource(
            source,
            loop=bool(self.get_parameter("loop").value),
            width=int(self.get_parameter("width").value) or None,
            height=int(self.get_parameter("height").value) or None,
        )
        # Depth 1: a slow subscriber should get the newest frame, not a backlog.
        self.publisher = self.create_publisher(Image, topic, 1)
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.meter = RateMeter()
        self.log_timer = self.create_timer(LOG_EVERY_S, self.log_rate)
        self.done = False
        size = self.source.frame_size
        self.get_logger().info(
            f"Publishing {self.source.kind} {source!r} on '{topic}' at {fps:g} fps"
            + (f" ({size[0]}x{size[1]})" if size else "")
        )

    def publish_frame(self) -> None:
        frame, timestamp = self.source.read()
        if frame is None:
            self.get_logger().info(f"Source ended after {self.meter.total} frames")
            self.timer.cancel()
            self.done = True
            return
        self.publisher.publish(frame_to_image_msg(frame, timestamp, Image, self.frame_id))
        self.meter.tick()

    def log_rate(self) -> None:
        self.get_logger().info(f"Publishing at {self.meter.take():.1f} Hz")

    def destroy_node(self) -> None:
        self.source.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
