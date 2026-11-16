# integration-team

**Members:** Phil, Henry (@ThunderDinInteractive) (shared with testing-deployment-team)

Owns calibration, geometry, and ROS 2 packaging: turning detection-team's
image-space `Detection`s into robot-relative `PerceptionObject`s (see
`shared/types.py`) and publishing them for the rest of the robot stack.
See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full pipeline.

## Semester roadmap

Progress markers reflect what's visible in the repo as of Oct 1.

### September — Fundamentals & setup

- **Week 1** *(in progress: ROS 2 image chosen; workspace not in the repo yet)* —
  Install ROS 2 (ros:jazzy-perception) and get a workspace building. Also install
  the root [requirements.txt](../requirements.txt) so you can read
  detection-team's output types.
- **Week 2** — Review ROS 2 image/message concepts: topics, message
  types, nodes, publishers/subscribers. Camera calibration basics: what
  intrinsics/extrinsics and lens distortion actually are, conceptually,
  before writing any calibration code.
- **Week 3** — Get a minimal ROS 2 publisher/subscriber pair running
  locally end to end. *End-of-September milestone (team-wide): everyone
  can load/manipulate images with OpenCV and run an existing detector.*

### October — Prep for calibration & geometry

Detection is the main focus this month elsewhere in the team, so use
October to get ready for November's calibration/geometry push rather
than sitting idle:

- **Week 4** — Read up on coordinate systems and perspective geometry
  (image plane vs. world/ground plane) — this is the math `geometry/`
  will implement in November.
- **Week 5** — Study OpenCV's camera calibration API
  (`cv2.calibrateCamera`, checkerboard detection) and sketch
  `calibration/calibrate.py`'s structure.
- **Week 6** — Look at `shared/types.py`'s `Detection` and
  `PerceptionObject` and sketch what the perception message schema
  (`messages/`) should look like; print/prepare a checkerboard
  calibration target so you're ready to capture calibration images in
  week 7. *End-of-October milestone (team-wide): a reproducible ball
  detector on unseen footage.*

### November — Perception beyond bounding boxes

- **Week 7** — Calibrate a test camera: capture checkerboard images,
  run `calibration/calibrate.py`, and undistort frames with OpenCV.
- **Week 8** — Implement ground-plane projection (`geometry/projection.py`):
  determine the center/base point of a detection's bounding box and
  project it onto the ground plane using the calibration from week 7.
- **Week 9** *(nodes written and tested without ROS, Nov 16; not yet run
  inside ROS 2: see [ROS 2 nodes](#ros-2-nodes))* — Define the perception
  message schema
  (`messages/perception_message.py`), start the first ROS 2 nodes for
  camera input (`ros2_nodes/camera_node.py`) and detections, and wire it
  together into a prototype: detection-team's ball detector → projection
  → perception message. *End-of-November milestone (team-wide): a
  prototype pipeline detects the ball and converts it to an approximate
  robot-relative position.*

### December — Documentation & integration

- **Week 10** — Harden the calibration/projection math based on
  testing-deployment-team's accuracy validation.
- **Week 11** — Package the trained detector as a clean ROS 2
  node/module (`ros2_nodes/perception_node.py`), combining detection,
  projection, and message publishing; document the calibration
  procedure, coordinate conventions, and message schema in this README
  so a new member could reproduce the setup.
- **Week 12** — Support final perception demo prep. *End-of-semester
  milestone (team-wide): a working, documented RoboCup perception
  prototype.*

## Folder layout

```
integration-team/
├── calibration/   # camera intrinsics, distortion correction
├── geometry/      # coordinate transforms, ground-plane projection
├── ros2_nodes/    # ROS 2 nodes (camera input, detection, projection)
├── messages/      # conversion between shared types and the ROS 2 messages
├── ros_ws/        # colcon workspace; the .msg files are in src/my_robot_controller/msg/
└── README.md
```

## ROS 2 nodes

Two nodes in `ros2_nodes/` run the pipeline end to end:

```
camera_node ──camera/image_raw──> perception_node ──perception/objects──> robot stack
(sensor_msgs/Image, bgr8)         (Detector + projection)   (my_robot_controller/msg/PerceptionObject)
```

| File | What it does |
|---|---|
| `camera_node.py` | Publishes frames from a video file, a folder of images, or a camera index. |
| `perception_node.py` | Runs `multiclass_v2` on each frame, projects every detection onto the floor and publishes one `PerceptionObject` message per object. |
| `pipeline.py` | All the logic both nodes use, with no ROS in it: `FrameSource`, the image-message conversion and `PerceptionPipeline`. Also runs the pipeline from the command line without ROS. |

**Try it without ROS** (any laptop with the root `requirements.txt`):

```bash
python integration-team/ros2_nodes/pipeline.py \
    --source path/to/video.mp4 --weights detection-team/models/multiclass_v2.pt \
    --params camera_params.yaml
```

It prints projected positions and the frame rate against the 10 Hz
target. `--params` is `calibrate.py`'s output with a `mount:` section
(`calibration/CALIBRATION_GUIDE.md`, Step 6).

**Run the nodes** (inside ROS 2 Jazzy, with the root `requirements.txt`
installed in the Python that ROS uses):

```bash
cd integration-team/ros_ws && colcon build && source install/setup.bash && cd ../..

# terminal 1: frames from a file now; later, the camera: -p source:="'0'" -p width:=1920 -p height:=1200
python3 integration-team/ros2_nodes/camera_node.py --ros-args \
    -p source:=/path/to/video.mp4 -p loop:=true

# terminal 2
python3 integration-team/ros2_nodes/perception_node.py --ros-args \
    -p weights:=detection-team/models/multiclass_v2.pt -p camera_params:=camera_params.yaml

# terminal 3: watch the output and its rate
ros2 topic echo /perception/objects
ros2 topic hz /camera/image_raw
```

Each node's docstring lists its parameters. Things to know:

- **One message per object.** The schema has no list message, so objects
  from one frame share a `timestamp` (the frame's capture time, Unix
  seconds) and a frame with nothing on the floor publishes nothing.
  Whether the robot stack wants a per-frame list message instead is an
  open question for the team.
- **Wrong resolution is an error.** If frames aren't the size the camera
  was calibrated at, the perception node stops with a message instead of
  publishing wrong positions.
- **The nodes are run as scripts**, not installed by `colcon`: the
  `my_robot_controller` package only builds the messages.

**What has and hasn't been run (Nov 16).** No machine used so far has
ROS 2, and CI doesn't either. Tested without ROS
(`testing-deployment-team/tests/test_ros2_nodes.py`): frame sources, image
conversion, detect + project against a synthetic camera, and both nodes
wired together through a fake `rclpy`. `pipeline.py` was also run on real
test-set footage with `multiclass_v2` on a MacBook CPU (M3 Max): 26–28
frames/s with the `.pt`, 42–46 with the dynamic ONNX export, at 320x240
and at 1920x1200, video decoding included. **Not yet done:** running the
two nodes inside ROS 2 (the real `rclpy`, `colcon build` of the messages,
image transport between processes, `ros2 topic hz`), and anything on the
robot's computer. The first run inside ROS 2 should expect small fixes.

## Setup

Uses the root [requirements.txt](../requirements.txt) plus ROS 2 itself
(install per your OS/distro — not pip-installable). Add any
integration-specific Python packages (e.g. `rclpy` extras) to a
`requirements.txt` in this folder once the Nov ROS 2 work starts.

## First task

Get a ROS 2 workspace building and a minimal publisher/subscriber pair
running locally — that's the Sept environment-setup milestone this
subteam's Nov work depends on.
