# integration-team

**Members:** Phil

Owns calibration, geometry, and ROS 2 packaging: turning detection-team's
image-space `Detection`s into robot-relative `PerceptionObject`s (see
`shared/types.py`) and publishing them for the rest of the robot stack.
See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full pipeline.

## Semester roadmap

### September — Fundamentals & setup

- **Week 1** — Install ROS 2 (ros:jazzy-perception) and get a workspace building. Also install
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
- **Week 9** — Define the perception message schema
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
├── messages/      # perception message definitions
└── README.md
```

## Setup

Uses the root [requirements.txt](../requirements.txt) plus ROS 2 itself
(install per your OS/distro — not pip-installable). Add any
integration-specific Python packages (e.g. `rclpy` extras) to a
`requirements.txt` in this folder once the Nov ROS 2 work starts.

## First task

Get a ROS 2 workspace building and a minimal publisher/subscriber pair
running locally — that's the Sept environment-setup milestone this
subteam's Nov work depends on.
