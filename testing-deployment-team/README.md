# testing-deployment-team

**Members:** Hossein Hatami Yazd, Henry (@ThunderDinInteractive) (shared with integration-team)

Owns evaluation, robustness testing, benchmarking, optimization research,
and pulling the semester's work into one documented, reproducible result.
See [ARCHITECTURE.md](../ARCHITECTURE.md) for how this touches both
detection-team and integration-team's output.

## Semester roadmap

Progress markers reflect what's visible in the repo as of Oct 1.

### September — Fundamentals & setup

- **Week 1** — Install the root [requirements.txt](../requirements.txt);
  set up the environment you'll use to run other teams' models.
- **Week 2** — Learn how bounding boxes and confidence scores work, and
  the definitions of precision, recall, and mAP — you'll be computing
  these for real starting in October. Run an existing pretrained
  detector yourself and manually inspect its output.
- **Week 3** *(done)* — Sketch the evaluation harness's interface (what it takes
  in, what it reports) so it's ready to point at detection-team's first
  model. *End-of-September milestone (team-wide): everyone can
  load/manipulate images with OpenCV and run an existing detector.*

### October — First RoboCup detection system

- **Week 4** *(done: precision, recall, mAP@0.5 and ms/frame, running in CI)* —
  Build the evaluation harness (`tests/eval_harness.py`):
  given weights and a dataset split, compute precision/recall/mAP.
  Coordinate with detection-team's `training/eval.py` so you're not
  building two versions of the same thing.
- **Week 5** — Run the harness against detection-team's first trained
  ball model as soon as it exists; report results back to them.
- **Week 6** — Test the detector on footage it's never seen; log
  specific failure cases (missed detections, false positives) for
  data-team and detection-team to act on, and document the evaluation
  procedure. *End-of-October milestone (team-wide): a reproducible ball
  detector on unseen footage.*

### November — Perception beyond bounding boxes

- **Week 7** — Re-run the evaluation harness as detection-team's model
  expands to robots/goalposts/field lines/landmarks.
- **Week 8** *(tool ready: `benchmarks/projection_accuracy.py`, see
  [Projection accuracy check](#projection-accuracy-check); waiting on the
  real camera for a real run)* — Help validate ground-plane projection accuracy: compare
  integration-team's projected robot-relative positions against known
  reference points.
- **Week 9** — Continue validating calibration/projection accuracy as
  integration-team's geometry code stabilizes, and test the end-to-end
  prototype (detect → project), reporting accuracy of the resulting
  robot-relative position. *End-of-November milestone (team-wide): a
  prototype pipeline detects the ball and converts it to an approximate
  robot-relative position.*

### December — Documentation & integration

- **Week 10** — Test detection reliability against lighting changes,
  motion blur, camera angle, partial occlusion, and different
  fields/backgrounds (`tests/`).
- **Week 11** — Benchmark inference speed (`benchmarks/inference_speed.py`);
  investigate ONNX export (`deployment/export_onnx.py`) and, if the
  future onboard computer is NVIDIA-based, begin studying TensorRT.
- **Week 12** — Compile final documentation (`docs/`) covering training,
  testing, dataset preparation, and inference, so a new member could
  reproduce the whole system, and prep the end-of-semester perception
  demo. *End-of-semester milestone (team-wide): a working, documented
  RoboCup perception prototype.*

## Folder layout

```
testing-deployment-team/
├── tests/         # evaluation harness, robustness test cases
├── benchmarks/    # inference speed / accuracy benchmarks
├── deployment/    # ONNX/TensorRT export experiments
├── docs/          # final write-up, reproduction guide
└── README.md
```

## Setup

Uses the root [requirements.txt](../requirements.txt). Dec work will need
`onnx` (and `onnxruntime`) for export, and `tensorrt` only if targeting an
NVIDIA onboard computer — add those to a `requirements.txt` in this folder
when that work starts, rather than the shared one, since not everyone
needs them.

## Benchmark harness

`tests/eval_harness.py` is implemented and runnable today:

```bash
python testing-deployment-team/tests/eval_harness.py
```

With no arguments it runs the bundled `dummy_detector.py` (simple color
thresholding, no model) over the tiny synthetic dataset in
`tests/fixtures/sample_dataset/`, and:

1. Computes precision, recall and mAP (per-class AP, averaged over
   classes that have ground truth) at IoU 0.5, plus mean ms/frame.
2. Writes the results to `benchmarks/results/latest.json`.
3. Compares recall against `benchmarks/baseline.json` and exits nonzero
   if it dropped by more than `--max-recall-drop` (default `0.05`).

To point it at a real model and a real dataset instead:

```bash
python testing-deployment-team/tests/eval_harness.py \
    --detector-file detection-team/inference/detector.py \
    --detector-class Detector \
    --weights path/to/weights.pt \
    --manifest path/to/real/manifest.yaml
```

The manifest is a YAML or JSON list of `{image, annotations}` entries —
see the docstring at the top of `eval_harness.py` for the exact format,
and `tests/fixtures/generate_sample_dataset.py` for a worked example.
`--detector-file`/`--detector-class` load any class implementing
`predict(image) -> list[Detection]`, constructed with a `weights_path`
keyword it's free to ignore — the same interface
`detection-team/inference/detector.py`'s `Detector` has, so it plugs in
directly with no adapter code.

### Visual report: what it found vs. what it should have found

Add `--report-dir` to any harness run to also get an HTML page you can
open in a browser:

```bash
python testing-deployment-team/tests/eval_harness.py \
    --report-dir testing-deployment-team/benchmarks/results/report
open testing-deployment-team/benchmarks/results/report/index.html   # macOS; or double-click it
```

Every frame is drawn with the ground truth (what the detector *should*
find) and the detector's predictions, colored by how the harness scored
them:

| On the image | Meaning |
|---|---|
| solid green box, `ball 0.87` | correct detection (true positive) |
| solid red box, `FP ball 0.42` | false alarm (false positive) |
| dashed orange box, `MISSED ball` | an expected object nothing matched (false negative) |
| thin white box | ground truth that was found (under its green box) |

The page shows the run's metrics up top, lists failures first, and has
filter buttons for missed / false alarms / correct frames. It uses the
harness's own matching (`match_frame_indices`), so every box on the page
is exactly what went into the precision/recall/mAP numbers. Anything
under `benchmarks/results/` is gitignored, so reports there stay local.

### Stand-in: pretrained COCO YOLO

Until detection-team has a trained checkpoint,
`tests/pretrained_yolo_detector.py` wraps the same stock COCO weights
`scripts/quickstart.py` uses, so the harness can be proven on a real
model and real images now (ARCHITECTURE.md handoff #2). COCO has no
RoboCup classes, so it only keeps `sports ball` → `ball` and drops the
rest. Its numbers are an interim baseline for the harness, not a
preview of our own model.

```bash
python testing-deployment-team/tests/eval_harness.py \
    --detector-file testing-deployment-team/tests/pretrained_yolo_detector.py \
    --detector-class PretrainedYoloDetector \
    --manifest path/to/real/manifest.yaml \
    --no-baseline-check
```

Needs `ultralytics` from the root `requirements.txt`. The manifest
needs real images with hand-checked ball boxes; a few dozen frames
from TORSO-21 (or our own footage) is enough for this purpose.

Once a real model exists, re-run the harness against a real held-out
set and overwrite `benchmarks/baseline.json` with those numbers (commit
that change on its own so it's easy to see when/why the baseline
moved).

## Projection accuracy check

`benchmarks/projection_accuracy.py` measures how far integration-team's
projected positions (`integration-team/geometry/projection.py`) are from
positions measured with a tape measure, and checks the milestone: error
≤ 0.3 m at 2 m range.

```bash
python testing-deployment-team/benchmarks/projection_accuracy.py \
    reference_points.csv --params camera_params.yaml
```

`camera_params.yaml` is `calibrate.py`'s output plus a `mount:` section
(see `integration-team/calibration/CALIBRATION_GUIDE.md`, Step 6).

**Setting up the rig.** Put the camera at the robot's camera height and
tilt. Mark points on the floor with tape and measure each one from the
robot's origin, which is the floor directly under the lens if the mount's
`x_m`/`y_m` are 0. +x is forward and +y is left, in meters. Get several
points near 2 m, some to the sides, plus nearer and farther ones to see
how error grows with distance.

**Two ways to fill in the CSV** (columns are documented in the script's
docstring):

| Mode | Each row has | What it tests |
|---|---|---|
| Pixel | `name,x_m,y_m,u,v`: the tape mark's pixel in one photo, read in any image viewer that shows cursor coordinates | Calibration, mount measurement and projection math |
| Detection | `name,x_m,y_m,image`: a photo of a ball placed on the mark (needs `--weights`) | The full detect → project prototype (Week 9) |

It prints every point's error, error by distance band, and a PASS/FAIL
line. It writes JSON to `benchmarks/results/projection_accuracy.json`
(gitignored), and exits nonzero unless the check passes. The check needs
at least one point within 0.25 m of 2 m, and fails if any of those points
is more than 0.3 m off or can't be projected. `--target-error`,
`--target-range` and `--target-band` change those numbers.

If error is consistently in one direction (all too far, or all too near),
suspect the mount's pitch or height before the projection code. See the
"If the numbers look wrong" table at the top of `projection.py`.

## First task

Sketch the evaluation harness: given a trained model and a val split, run
inference and report precision/recall/mAP — mirroring what detection-team
needs for its Oct milestone. Coordinate with detection-team so you're not
building two versions of the same thing.
