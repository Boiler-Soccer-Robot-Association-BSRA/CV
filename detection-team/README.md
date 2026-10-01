# detection-team

**Members:** Suhaas, Aditya Mitra

Owns model training and the core object detector. Produces the
`Detection` objects (see `shared/types.py`) that integration-team
projects into robot-relative positions — see
[ARCHITECTURE.md](../ARCHITECTURE.md).

## Semester roadmap

Progress markers reflect what's visible in the repo as of Oct 1.

### September — Fundamentals & setup

- **Week 1** *(in progress: sample test image added in `training/test-images/`)* —
  Install and verify the OpenCV/PyTorch/Ultralytics
  environment (root `requirements.txt`); run a pretrained YOLO model on
  a sample image as a smoke test.
- **Week 2** — OpenCV fundamentals: images, pixels, color spaces,
  filtering, contours, transformations. Write a few basic OpenCV
  programs, then move on to intro CNNs/object detection, intro YOLO,
  and PyTorch basics — including how bounding boxes and confidence
  scores work.
- **Week 3** — Write basic bounding-box + confidence visualization
  scripts against the pretrained model's output. *End-of-September
  milestone (team-wide): everyone can load/manipulate images with
  OpenCV and run an existing detector.*

### October — First RoboCup detection system

- **Week 4** — Once data-team has a first train/val/test split, review
  it and set up a training config (`configs/`) for ball detection.
- **Week 5** — Train the first small YOLO model on ball detection
  (`training/train.py`).
- **Week 6** — Measure precision/recall/mAP and inference speed
  (`training/eval.py`); test the detector on footage it hasn't seen and
  share failure cases with data-team; move inference out of notebooks
  into `inference/` as a reusable module (`inference/detector.py`); and
  document the training config and procedure. *End-of-October milestone
  (team-wide): a reproducible ball detector on unseen footage.*

### November — Perception beyond bounding boxes

- **Week 7** — Once data-team's expanded labels start coming in, begin
  training on robots and goalposts in addition to the ball
  (`shared/classes.py`).
- **Week 8** — Extend training to field lines and landmarks; re-run
  eval across all classes.
- **Week 9** — Share detection output format with integration-team as
  they define the perception message schema and support their first
  ROS 2 node work with sample `Detection` outputs; tune the ball model
  specifically for reliability, since it's the one going into the Nov
  prototype pipeline. *End-of-November milestone (team-wide): a
  prototype pipeline detects the ball and converts it to an approximate
  robot-relative position.*

### December — Documentation & integration

- **Week 10** — Work with testing-deployment-team on reliability issues
  found under lighting changes, motion blur, camera angle, occlusion,
  and background variation.
- **Week 11** — Support ONNX export of trained weights for
  testing-deployment-team's benchmarking/deployment work; write up the
  full training procedure (config, dataset version, hyperparameters,
  how to reproduce a run) in this README.
- **Week 12** — Support final perception demo prep. *End-of-semester
  milestone (team-wide): a working, documented RoboCup perception
  prototype.*

## Folder layout

```
detection-team/
├── configs/      # training configs (model, hyperparameters, dataset paths)
├── training/     # train.py, eval.py
├── inference/    # reusable inference module (no notebook-only code)
├── notebooks/    # exploratory training/eval notebooks
├── models/       # trained weights (gitignored, do not commit)
└── README.md
```

## Setup

Uses the root [requirements.txt](../requirements.txt) (ultralytics,
opencv-python, torch, numpy) — no additional dependencies expected.

## First task

Run a pretrained YOLO model (via `ultralytics`) on a sample image or
webcam frame and confirm the environment works end to end — that's the
Sept baseline everything else this semester builds on.
