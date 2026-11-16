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

- **Week 10** *(lighting, motion blur and occlusion done early, Nov 16:
  see [Robustness tests](#robustness-tests); camera angle and other
  fields still need real footage)* — Test detection reliability against lighting changes,
  motion blur, camera angle, partial occlusion, and different
  fields/backgrounds (`tests/`).
- **Week 11** *(speed benchmark done early, Nov 16: see
  [Inference speed](#inference-speed))* — Benchmark inference speed
  (`benchmarks/inference_speed.py`);
  investigate ONNX export (`deployment/export_onnx.py`; *done early,
  Nov 16: see [ONNX export](#onnx-export)*) and, if the
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

Uses the root [requirements.txt](../requirements.txt). ONNX work
(running or exporting `.onnx` weights) also needs this folder's
[requirements.txt](requirements.txt):

```bash
pip install -r testing-deployment-team/requirements.txt
```

It's kept out of the shared file since not everyone needs it. `tensorrt`
would go there too, only if the onboard computer turns out to be NVIDIA.

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
3. Compares recall against `benchmarks/baseline_dummy.json` and exits
   nonzero if it dropped by more than `--max-recall-drop` (default `0.05`).

That run proves the harness works end to end with no model. The gate on
the real model is the next section.

### Real-model regression gate

`benchmarks/baseline.json` holds `multiclass_v2`'s scores on the held-out
test set (multiclass_r180, 1,405 frames, conf 0.20, IoU 0.5, recorded Nov
16 on a MacBook CPU): **0.853 recall / 0.834 precision, mAP 0.827**; recall
per class as scored: ball 0.790, goalpost 0.864, landmark 0.857, robot
0.876.

CI can't run the model, since weights and TORSO-21 aren't in git. So the
model's detections on the test set are recorded once on a laptop that has
both, and committed as two small JSON files in `benchmarks/recorded/`
(boxes only: the split's labels, and what the model found). CI scores them
with the same code as a live run:

```bash
python testing-deployment-team/tests/eval_harness.py --recorded
```

It fails if recall, overall **or for any class**, is more than 0.05 below
the baseline.

| It catches | It doesn't catch |
|---|---|
| A change to the scoring code (matching, AP, recall) that moves the real model's numbers | A broken `Detector`, preprocessing change or dependency upgrade: CI runs no inference |
| A new model whose recording scores worse than the baseline, overall or in one class | Anything about speed (`ms_per_frame` is stored, not gated) |
| A recording made on a different split than the labels (fingerprints must match) | |

To record a model (new weights, a changed test split, or after touching
`Detector` or upgrading ultralytics):

```bash
python testing-deployment-team/tests/record_detections.py \
    detection-team/models/multiclass_v2.pt \
    data-team/datasets/multiclass_r180/data.yaml --conf 0.20
python testing-deployment-team/tests/eval_harness.py --recorded \
    testing-deployment-team/benchmarks/recorded/multiclass_v2_multiclass_r180_test_detections.json
```

Add `--write-baseline` to the first command only when the new scores
should become the baseline (a new model for the robot, or new labels), and
commit that baseline change on its own so it's easy to see when and why
it moved. `record_detections.py`'s docstring covers each case.

The baseline is recall **as scored**. The count-once rule for recording
151's ball (ball 0.907) isn't applied in the gate.

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

A real model exists now, so `benchmarks/baseline.json` comes from it: see
[Real-model regression gate](#real-model-regression-gate).

## Robustness tests

`tests/robustness.py` re-runs a model over a test split once per
*condition*, applying the condition to every frame, and reports recall per
class next to the clean result:

```bash
python testing-deployment-team/tests/robustness.py \
    detection-team/models/multiclass_v2.pt \
    data-team/datasets/multiclass_r180/data.yaml --conf 0.20 --count-once 151-:ball
```

Lighting and motion blur reuse data-team's transforms
(`data-team/scripts/augment.py`); occlusion (a gray patch over part of each
labeled object) is new here. Scoring is the eval harness's matching.
`--examples-dir` saves sample frames per condition, and the script's
docstring has every setting.

**Measured Nov 16:** `multiclass_v2.pt`, multiclass_r180 test set (1,405
frames, 6,806 objects), conf 0.20, IoU 0.5, MacBook CPU. Recall, with the
change from clean in points. "Ball, counted once" applies the team's rule
for recording 151's stationary ball; plain "ball" is as scored.

| Condition | What changes | ball | ball, counted once | goalpost | landmark | robot | all |
|---|---|---|---|---|---|---|---|
| clean | nothing | 0.790 | 0.907 | 0.864 | 0.857 | 0.876 | 0.853 |
| dim | contrast x0.8, brightness -30 | 0.755 (-3.5) | 0.868 (-4.0) | 0.794 (-7.0) | 0.804 (-5.2) | 0.800 (-7.6) | 0.796 (-5.7) |
| dark | contrast x0.5, brightness -40, gamma 1.5 | 0.441 (-34.9) | 0.506 (-40.1) | 0.232 (-63.2) | 0.331 (-52.6) | 0.218 (-65.8) | 0.310 (-54.3) |
| bright | contrast x1.2, brightness +40, gamma 0.7 | 0.775 (-1.5) | 0.890 (-1.7) | 0.876 (+1.2) | 0.852 (-0.5) | 0.854 (-2.2) | 0.846 (-0.6) |
| low contrast | contrast x0.5, brightness +64 | 0.760 (-3.0) | 0.873 (-3.4) | 0.874 (+1.0) | 0.844 (-1.2) | 0.827 (-4.9) | 0.835 (-1.7) |
| warm cast | red +25%, blue -25% | 0.800 (+1.0) | 0.919 (+1.2) | 0.890 (+2.6) | 0.842 (-1.4) | 0.793 (-8.3) | 0.836 (-1.7) |
| cool cast | blue +25%, red -25% | 0.758 (-3.2) | 0.870 (-3.7) | 0.853 (-1.1) | 0.843 (-1.4) | 0.765 (-11.1) | 0.820 (-3.3) |
| motion blur 5 px | horizontal | 0.769 (-2.1) | 0.883 (-2.4) | 0.854 (-1.0) | 0.871 (+1.4) | 0.836 (-4.0) | 0.849 (-0.3) |
| motion blur 9 px | horizontal | 0.691 (-9.9) | 0.793 (-11.4) | 0.845 (-1.9) | 0.857 (+0.0) | 0.764 (-11.2) | 0.818 (-3.5) |
| motion blur 15 px | horizontal | 0.611 (-17.9) | 0.702 (-20.5) | 0.771 (-9.3) | 0.791 (-6.6) | 0.626 (-25.0) | 0.736 (-11.6) |
| 25% occluded | gray patch on a random side of each object | 0.642 (-14.8) | 0.738 (-17.0) | 0.689 (-17.5) | 0.634 (-22.3) | 0.808 (-6.8) | 0.674 (-17.9) |
| 50% occluded | same, half of each object | 0.368 (-42.2) | 0.423 (-48.5) | 0.526 (-33.8) | 0.390 (-46.6) | 0.245 (-63.1) | 0.383 (-46.9) |

What this says, for detection-team (handoff #8 in ARCHITECTURE.md):

- **Brighter, flatter or tinted light costs little** (overall within 3.3
  points). The exception is **robots under a color cast**: -8 points warm,
  -11 cool.
- **Dimmer light costs more than brighter.** A mild dim drops every class
  3 to 8 points, enough to put goalpost (0.794) just under the 0.80
  target. The `dark` condition is deliberately extreme (frames are nearly
  black) and recall falls to 0.31; the model keeps few false alarms there
  (precision 0.90), so it goes quiet instead of guessing.
- **Motion blur hurts ball and robot most.** 5 px is nearly free; at 9 px
  ball and robot lose 10 to 11 points while goalposts and landmarks barely
  move; at 15 px robot is down 25 points and ball 18.
- **Occlusion is the biggest weakness measured**, even at 25% (-18 points
  overall; robots hold up best there, -7). Read these two rows with care:
  a detection must overlap the *full* labeled box by IoU 0.5, so a tight
  box around only the visible half counts as a miss and a false alarm.
  Part of the drop is that scoring rule, not the object going unseen
  (precision falls to 0.67 and 0.47, which fits). For projection that
  still matters: a robot boxed without its feet lands in the wrong place.
- Ideas, not yet tried: train with stronger darkening, blur and
  cutout-style augmentation (`augment.py` has the first two; no training
  was run for this report).

**Limits.** These are synthetic changes to the existing test frames, not
new footage. Blur is in pixels on frames of 320x240 to 800x600, so the
same shake is more pixels on the 1920x1200 camera. A flat gray patch is a
cruder occluder than a robot's leg. One run per condition; differences
under about 3 points are within the run-to-run noise seen between
trainings. Camera angle and other fields/backgrounds (also on Week 10's
list) can't be simulated this way.

## Inference speed

`benchmarks/inference_speed.py` times a model on CPU and reports avg,
median and p95 milliseconds per frame, and FPS. It takes `.pt` and `.onnx`
weights (ONNX needs this folder's `requirements.txt`):

```bash
python testing-deployment-team/benchmarks/inference_speed.py \
    detection-team/models/multiclass_v2.pt detection-team/models/multiclass_v2.onnx \
    --images data-team/datasets/multiclass_r180/images/test --runs 300
```

Each run is one `Detector.predict()` call: resize, inference, NMS and
conversion to `Detection`s. Reading the image from disk isn't timed, and
neither are the first `--warmup` calls (default 10). Use real test frames
with `--images` for any number you report: the default is the tiny
synthetic fixture frames, which only shows the script runs. Results are
also written to `benchmarks/results/inference_speed.json` (gitignored).

**Measured Nov 16** on a MacBook Pro (Apple M3 Max, CPU only, 8 torch
threads; torch 2.14.0, onnxruntime 1.31.0). 300 frames spread evenly over
the multiclass_r180 test set (320x240 to 800x600), 300 timed runs after 20
warmup, conf 0.20, 640 px model input:

| Weights | Avg ms | Median ms | p95 ms | FPS |
|---|---|---|---|---|
| `multiclass_v2.pt` | 28.7 | 29.3 | 32.1 | 34.9 |
| `multiclass_v2.onnx` | 22.2 | 22.2 | 22.9 | 45.0 |
| `ball_v4.pt` | 29.0 | 29.7 | 32.0 | 34.5 |
| `ball_v4.onnx` | 22.0 | 21.9 | 22.8 | 45.5 |

A second run right after, in the opposite order, gave 31.8 / 44.3 FPS for
`multiclass_v2` (`.pt` / `.onnx`) and 33.7 / 43.8 for `ball_v4`, so read
these as **about 32–35 FPS for `.pt` and 44–45 FPS for ONNX**, not to the
decimal. That agrees with the eval-harness speeds in
[detection-team's results](../detection-team/README.md#results) (31–32 FPS
`.pt`, 43–44 ONNX), which also count a few slower first frames.

- ONNX is about 25–30% faster on this CPU and steadier (p95 within 1 ms of
  the median).
- The four-class model costs the same as the ball-only one: both are
  YOLOv8n and only the last layer differs.
- Both clear the ≥10 Hz ROS 2 target with room to spare **on this laptop**.
  The robot's onboard computer will be slower; re-run this there before
  relying on it.

## ONNX export

`deployment/export_onnx.py` exports `.pt` weights to ONNX at the image
size the model was trained at (read from the checkpoint; 640 for ours),
and with `--check` scores the export against the `.pt` on a test set using
the same code as `detection-team/training/eval.py`:

```bash
python testing-deployment-team/deployment/export_onnx.py \
    detection-team/models/multiclass_v2.pt \
    --output detection-team/models/multiclass_v2_dynamic.onnx --dynamic \
    --check data-team/datasets/multiclass_r180/data.yaml --conf 0.20
```

It exits nonzero if precision or recall differ from the `.pt` by more than
`--max-diff` (default 0.02). An `.onnx` is a model weight: it's gitignored
and shared through the team drive, never committed.

**Fixed or dynamic input.** Without `--dynamic` the file takes one fixed
1x3x640x640 input, so every frame is padded to a square. With `--dynamic`
it accepts any size and gets the same padding the `.pt` gets (a 640x480
frame runs as 640x480), so it does less work per frame and its detections
match the `.pt`'s.

**Measured Nov 16**, MacBook Pro (M3 Max, CPU, onnxruntime 1.31.0), conf
0.20, IoU 0.5, full test sets. FPS is from `inference_speed.py` (300 runs
on multiclass_r180 test frames, same session for all six rows):

| Weights | Test set | Precision | Recall | mAP@0.5 | FPS |
|---|---|---|---|---|---|
| `multiclass_v2.pt` | multiclass_r180 (1,405 frames, 6,806 objects) | 0.834 | 0.853 | 0.827 | 32.8 |
| ONNX, fixed 640x640 | same | 0.822 | 0.850 | 0.823 | 42.2 |
| **ONNX, `--dynamic`** | same | **0.834** | **0.853** | **0.827** | **54.7** |
| `ball_v4.pt` | ball_r180 (945 frames, 867 balls) | 0.921 | 0.803 | 0.800 | 34.4 |
| ONNX, fixed 640x640 | same | 0.916 | 0.810 | 0.806 | 45.3 |
| **ONNX, `--dynamic`** | same | **0.921** | **0.803** | **0.800** | **59.2** |

- **The hand-measured ONNX numbers hold.** The fixed export reproduces
  what detection-team's README reports for `multiclass_v2` (0.850 recall /
  0.822 precision, about 44 FPS) and for `ball_v4` (about 43 FPS).
- **The small accuracy gap was the square padding, not the export.** The
  dynamic export gives the same counts as the `.pt` (6,961 predictions on
  multiclass_r180, 756 on ball_r180).
- **Dynamic is also faster here**, because most test frames are wider than
  tall and it skips the padding rows. On a 1920x1200 frame (the See3CAM's
  full resolution, tested with one upscaled test image) `multiclass_v2`
  ran at 34 FPS as `.pt`, 44 fixed and 60 dynamic.
- **Recommendation: use the `--dynamic` export on the robot's CPU.** One
  caveat: it hasn't been run on the onboard computer, and some
  accelerators (TensorRT among them) prefer fixed shapes, so keep the
  fixed export as the fallback until that's known.

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

**Ready-made layout.** [`benchmarks/floor_grid/`](benchmarks/floor_grid/README.md)
has a 13-point taped grid (0.5 to 2.5 m ahead, up to 1 m to each side),
step-by-step setup, and CSV templates with the positions already filled
in for both modes below. Use it unless you need a different layout.

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
