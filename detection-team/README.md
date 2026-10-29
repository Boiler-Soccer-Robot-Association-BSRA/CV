# detection-team

**Members:** Suhaas, Aditya Mitra

Owns model training and the core object detector. Produces the
`Detection` objects (see `shared/types.py`) that integration-team
projects into robot-relative positions — see
[ARCHITECTURE.md](../ARCHITECTURE.md).

## Semester roadmap

Progress markers reflect what's visible in the repo as of Oct 18.

### September — Fundamentals & setup

- **Week 1** *(done: `./scripts/setup.sh` + `scripts/quickstart.py`
  verified on Python 3.12 / Apple Silicon, smoke test passes on
  `training/test-images/street_test.jpg`)* — Install and verify the OpenCV/PyTorch/Ultralytics
  environment (root `requirements.txt`); run a pretrained YOLO model on
  a sample image as a smoke test.
- **Week 2** *(individual learning — not visible in the repo)* —
  OpenCV fundamentals: images, pixels, color spaces,
  filtering, contours, transformations. Write a few basic OpenCV
  programs, then move on to intro CNNs/object detection, intro YOLO,
  and PyTorch basics — including how bounding boxes and confidence
  scores work.
- **Week 3** *(done: `inference/visualize.py`, see
  [Visualizing detections](#visualizing-detections))* — Write basic bounding-box + confidence visualization
  scripts against the pretrained model's output. *End-of-September
  milestone (team-wide): everyone can load/manipulate images with
  OpenCV and run an existing detector.*

### October — First RoboCup detection system

- **Week 4** *(done: data-team's split landed Oct 18; configs
  `ball_v2.yaml` / `ball_v2_aug.yaml` point at it — see
  [Training a model](#training-a-model))* — Once
  data-team has a first train/val/test split, review
  it and set up a training config (`configs/`) for ball detection.
- **Week 5** *(done: `ball_v2` trained on data-team's split, 0.79 test
  recall, short of 0.90 because of tiny balls; see [Results](#results))* — Train the
  first small YOLO model on ball detection (`training/train.py`).
- **Week 6** *(done: `training/eval.py`, `inference/detector.py`,
  [Results](#results) and [failure cases](#ball_v1-failure-cases-for-data-team))* —
  Measure precision/recall/mAP and inference speed
  (`training/eval.py`); test the detector on footage it hasn't seen and
  share failure cases with data-team; move inference out of notebooks
  into `inference/` as a reusable module (`inference/detector.py`); and
  document the training config and procedure. *End-of-October milestone
  (team-wide): a reproducible ball detector on unseen footage.*

### November — Perception beyond bounding boxes

- **Week 7** *(in progress: `multiclass_v1` training on Colab on the
  placeholder split; `configs/multiclass_v2.yaml` uses data-team's
  multi-class split — see [Training on Google Colab](#training-on-google-colab))* — Once
  data-team's expanded labels start coming in, begin
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
├── models/       # trained weights + provenance .json (gitignored, do not commit)
├── datasets/     # retired TORSO-21 placeholder datasets (gitignored)
├── runs/         # training runs, eval results/reports, annotated images (gitignored)
└── README.md
```

## Setup

Uses the root [requirements.txt](../requirements.txt) (ultralytics,
opencv-python, torch, numpy) — no additional dependencies expected.

## Visualizing detections

`inference/visualize.py` draws each detection's box and a
`class_name 0.87` confidence label on an image. Run it on one or more
images or folders; annotated copies go to `runs/visualize/` and a
per-image summary (class, confidence, box) is printed:

```bash
python detection-team/inference/visualize.py detection-team/training/test-images/
python detection-team/inference/visualize.py frame.jpg --conf 0.5          # hide low-confidence boxes
python detection-team/inference/visualize.py frame.jpg --weights best.pt   # a trained checkpoint
```

It defaults to the same stock COCO `yolov8n.pt` weights as
`scripts/quickstart.py`, so class names are COCO's (`person`,
`sports ball`, ...) until we point it at our own model. Trying a few
`--conf` values on the same image is a quick way to see what the
confidence threshold actually trades off.

From code, `draw_detections(image, detections)` takes any
`list[Detection]` — from the dummy detector, the pretrained model, or
`Detector` — which is handy for the Week 6 failure-case write-ups for
data-team.

## Using the detector

`inference/detector.py`'s `Detector` is the one way to run a model in
this repo: a BGR frame in, a `list[Detection]` out. integration-team's
ROS 2 node, the eval harness, `visualize.py` and `scripts/quickstart.py`
all go through it.

```python
import sys

sys.path.insert(0, "detection-team/inference")  # the hyphen means it can't be a package import
from detector import COCO_CLASS_MAP, Detector

detector = Detector("detection-team/models/ball_v1.pt")  # a model from training/train.py
detections = detector.predict(frame)  # frame: BGR uint8, (H, W, 3)
detections = detector.predict(frame, timestamp=capture_time)  # copied onto each Detection

coco = Detector("yolov8n.pt", class_map=COCO_CLASS_MAP)  # stock COCO: keep "sports ball" as "ball"
```

- `conf` (default 0.25) drops detections below that confidence;
  `device` (`"cpu"`, `"mps"`, `"cuda:0"`) picks hardware, default auto.
- A model trained with `training/train.py` already predicts our class
  names (`shared/classes.py`). Weights trained on another label set
  report *their* names unless you pass `class_map`, which renames the
  classes it lists and drops everything else.
- `timestamp` should be the frame's capture time; leave it `None` for
  static images, per the `Detection` contract in
  [ARCHITECTURE.md](../ARCHITECTURE.md#data-contracts).

## Training a model

Everything below runs from the repo root with the venv active. The
models in [Results](#results) were made exactly this way.

### 1. Get a dataset

Use data-team's split (seed 42, the same on every computer — see
[data-team/docs/splits-and-augmentation.md](../data-team/docs/splits-and-augmentation.md)):

```bash
python data-team/scripts/download.py --subset reality           # once, ~5.7 GB
python data-team/scripts/split.py                               # ball -> data-team/datasets/ball/
python data-team/scripts/augment.py data-team/datasets/ball     # optional: + data_aug.yaml
python data-team/scripts/split.py --classes ball robot goalpost landmark   # -> data-team/datasets/multiclass/
```

Check the `fingerprint` that `split.py` prints against a teammate's: the
same fingerprint means the same split, so your scores are comparable.

`training/placeholder_dataset.py` is **retired**: it built a temporary
split before data-team's existed and is kept only to reproduce `ball_v1`
and `multiclass_v1`.

### 2. Train

```bash
python detection-team/training/train.py detection-team/configs/ball_v2.yaml
```

A config (`configs/`) is the complete record of a run: starting
weights, dataset, epochs, image size, batch, seed. Copy one to a new
file with a new `name` per experiment instead of editing it. Training
writes `models/<name>.pt` and `models/<name>.json` — the JSON records
the config, the dataset's `split_info.yaml`, the git commit (and whether
the tree was dirty), library versions, and the final val metrics, so any
model can be traced back to exactly what made it. Ultralytics' own run
folder (loss curves, per-epoch CSV) is in `runs/train/<name>/`.

On an Apple Silicon Mac the GPU (`mps`) is picked automatically —
ultralytics alone would use the CPU, ~2.5× slower. One `ball_v1` epoch
took about 2.5 minutes on an M3 Max. Long runs make a laptop very hot —
use `--resume` to split a run up, or train on Colab (below). Weights aren't committed; share them via the
[team drive](https://drive.google.com/drive/folders/1qa2ktrauYvNnrdFLqBTY9tBebvrS4sXG).

### 3. Evaluate

```bash
python detection-team/training/eval.py detection-team/models/ball_v2.pt \
    data-team/datasets/ball/data.yaml --split test \
    --report-dir detection-team/runs/eval/ball_v2_test_report
open detection-team/runs/eval/ball_v2_test_report/index.html
```

This runs testing-deployment-team's eval harness with our `Detector`,
so both teams get the same numbers from the same code: precision and
recall at IoU 0.5 and confidence 0.25, mAP@0.5, and ms/frame.
`--report-dir` writes the visual report (every frame's ground truth vs.
predictions, colored correct / false alarm / missed — see
[testing-deployment-team's README](../testing-deployment-team/README.md#visual-report-what-it-found-vs-what-it-should-have-found)).
Add `--device cpu` to measure CPU speed, which is what the
[milestone](../docs/milestones.md) targets.

### Reproducing a run

Same config + same dataset (`split_info.yaml` lists every recording in
every split, and the builder is seeded) + the commit in the model's
JSON. Training on `mps` isn't bit-for-bit deterministic (PyTorch's Apple
GPU backend doesn't guarantee it), so a rerun lands close to, not
exactly on, the recorded metrics.

## Dataset contract

What detection-team's training and eval code expects from a dataset.
data-team's `split.py` produces exactly this (checked Oct 18), so
switching datasets is a one-line config change:

- **YOLO format.** A `data.yaml` with `path`, `train`, `val`, `test`
  (directories of images, relative to `path`) and `names`. Each image
  `images/<split>/<file>` has `labels/<split>/<file stem>.txt` with one
  `class_id cx cy w h` line per object, normalized to 0–1; an empty or
  missing file means "no objects".
- **Class ids are positions in [`shared/classes.py`](../shared/classes.py)'s
  `CLASSES`**: 0 ball, 1 robot, 2 goalpost, 3 field_line, 4 landmark.
  `names` must list every id from 0 up to the highest one used, even ones
  with no labels (YOLO needs them contiguous). This is what lets a
  model's output names line up with the shared contract with no mapping.
- **Split by recording / match / session, never by individual image.**
  Consecutive frames of one recording are near-duplicates, so a random
  per-image split puts copies of training frames in val/test and makes
  the model look far better on "unseen footage" than it is. Note that
  TORSO-21's own `test` set shares every one of its recordings with its
  `train` set, so it is **not** unseen footage either.
- **Fixed seed**, so the split is reproducible.
- **`landmark`** = field-line intersections (TORSO-21's L-, T- and
  X-Intersections, merged). They're labeled as points, so `split.py`
  boxes them as a square 5% of the image width centered on
  the point — the box *center* is what matters downstream.
- **`field_line`** isn't labeled with boxes; the team decided field lines
  stay out of the box detector (handled separately later).
- Optional but recommended: a `split_info.yaml` next to `data.yaml`
  describing how the split was made — `train.py` copies it into every
  model's provenance JSON.

## Results

**The October milestone is not met yet: speed is met, recall is not.**
`ball_v2` (trained on data-team's split) finds **79% of the balls** in
data-team's held-out test recordings at conf 0.20; the target is 90%.
Speed is fine: 30 FPS on a MacBook CPU, 42 FPS as ONNX. Almost all of
the gap is **one recording of tiny, far-away balls** (see
[ball_v2 failure cases](#ball_v2-failure-cases-for-data-team)): without
it, recall is 0.95; on the Humanoid League venues alone it is 0.97.

### ball_v2 (data-team's split)

data-team's **test** split (`split.py`, seed 42, fingerprint
`837ca5b99e04eaf8`): 8 TORSO-21 recordings `ball_v2` was not trained
on, 686 frames, 624 balls. Precision/recall at IoU 0.5; CPU speed on an
M3 Max with `--device cpu`.

| Model | Confidence cutoff | Precision | Recall | mAP@0.5 | CPU speed |
|---|---|---|---|---|---|
| `ball_v2` | 0.25 (default) | 0.906 | 0.788 | 0.786 | 30 FPS |
| **`ball_v2`** | **0.20** | **0.890** | **0.793** | **0.790** | **30 FPS** |
| `ball_v2` exported to ONNX | 0.20 | 0.894 | 0.800 | 0.795 | 42 FPS |
| `ball_v2`, test minus recording 151 | 0.20 | 0.924 | 0.949 | — | — |
| `ball_v2`, Humanoid League venues only | 0.20 | 0.977 | 0.968 | — | — |

- **Lowering the cutoff doesn't close the gap.** Recall by cutoff:
  0.05 → 0.849 (precision 0.71), 0.10 → 0.816, 0.20 → 0.793,
  0.30 → 0.780, 0.50 → 0.760 (precision 0.97). The missed balls get
  almost no score at all, so this needs a model change, not a setting.
- On the validation split (used to pick the checkpoint) it scored
  0.903 recall / 0.962 precision; test recall is lower because of
  recording 151.
- Trained on Colab (T4) for all 30 epochs; the kept checkpoint is epoch
  28. `models/ball_v2.json` records this, plus the Oct 23 incident
  where re-running the notebook overwrote the first saved copy with an
  empty model (fixed in #22; the weights used here are the original
  `runs/train/ball_v2/weights/best.pt`).
- Still use `conf=0.20` for ball models: it is the best balance here too.

#### ball_v2 failure cases (for data-team)

129 balls missed at conf 0.20:

1. **Tiny balls: 108 of 121 balls under 20 px wide are missed (89%)**,
   versus 4–5% for every bigger size. 103 of those are in a single test
   recording, **151** ("Nao Devils Lab", SPL), where every ball is tiny.
   That one recording is 80% of all misses. The venue itself isn't
   new: 5 training recordings come from it.
2. **Nao Devils Lab false alarms**: 53 of the 61 false alarms are
   there (the NAO robots' LED eyes again).
3. **Blur is not a big problem any more**: blurred/concealed balls are
   missed 12% of the time, clear ones 22% (the clear ones include the
   tiny balls above).
4. Next steps: train at a higher resolution (`ball_v3_960`, imgsz 960)
   so tiny balls cover more pixels, and/or more tiny-ball training
   data from data-team. `ball_v2_aug` (Aditya) tests whether
   augmentation helps.

### ball_v1 (placeholder split)

`ball_v1` was trained and tested on detection's temporary placeholder
split. On data-team's real split, half of the test recordings were in
`ball_v1`'s training data, so it can't be scored there fairly. On the 4
test recordings it never saw (304 balls) it reaches **0.77 recall / 0.72
precision** at conf 0.20, well below the 0.90 it scored on the
placeholder test.

The table below is all on the placeholder split's **test** set: 12
TORSO-21 recordings `ball_v1` was not trained on (1,685 frames, 1,160
balls). Precision/recall at IoU 0.5; CPU speed on an M3 Max with
`--device cpu`.

| Model | Confidence cutoff | Precision | Recall | mAP@0.5 | CPU speed |
|---|---|---|---|---|---|
| Stock `yolov8n.pt` (COCO "sports ball"), no training | 0.25 | 0.919 | 0.521 | 0.508 | — |
| `ball_v1` | 0.25 (default) | 0.827 | 0.898 | 0.888 | 33 FPS |
| **`ball_v1`** | **0.20** | **0.800** | **0.903** | **0.891** | **33 FPS** |
| `ball_v1` exported to ONNX | 0.20 | 0.808 | 0.916 | 0.904 | 42 FPS |

- **Use `conf=0.20` with `ball_v1`.** At that cutoff it reaches the
  milestone's numbers ([milestones](../docs/milestones.md): ≥90% recall
  at IoU 0.5 on held-out footage, ≥30 FPS on a MacBook CPU) on the
  placeholder test only. See the note above.
- Training is what made the difference: stock YOLO finds about half the
  balls, `ball_v1` finds 90%.
- **Precision is understated.** Some "false alarms" are real balls
  TORSO-21 never labeled (see failure cases below).
- `ball_v1` was trained for 16 of its 30 planned epochs, then stopped by
  hand: the laptop was overheating and val scores had plateaued since
  epoch 1. The kept checkpoint is epoch 10; `models/ball_v1.json`
  records this.
- ONNX is ~30% faster on CPU. Its accuracy differs slightly because the
  export pads every image to a fixed square; testing-deployment-team's
  `deployment/export_onnx.py` (December) is where that gets settled.

### multiclass_v1 (placeholder split)

Trained on Colab (T4 GPU, 40 epochs, resumed once after a disconnect) on
the retired placeholder multi-class split; tested on its 12 held-out
recordings (2,097 frames, 7,489 objects), conf 0.25, IoU 0.5:

| Class | AP@0.5 |
|---|---|
| ball | 0.863 |
| goalpost | 0.821 |
| robot | 0.765 |
| landmark | 0.729 |
| **All classes** | **precision 0.837, recall 0.786, mAP 0.794** |

- Close to the November target (≥80% recall across classes) already;
  landmarks (field-line crossings) are the weakest class.
- Same caveat as `ball_v1`: this is the placeholder split, so the real
  numbers come from `multiclass_v2` on data-team's split.
- The model, its `.json` and a visual report are on the Colab runner's
  Google Drive (`BSRA-CV/`); share the `.pt` via the team drive.

#### ball_v1 failure cases (for data-team)

From the visual report (`eval.py ... --report-dir`) and a breakdown of
every missed ball. Ordered by how much fixing each would help:

1. **Tiny, far-away balls are the biggest miss.** Balls under ~20 px
   wide (in a 640-px-wide frame) are missed 44% of the time (67 of
   153), versus 2–6% for anything bigger. More training images with
   distant balls would help most.
2. **One venue is much worse: "Nao Devils Lab" (SPL league) — 50% of
   balls missed.** Its balls are mostly tiny and right at the frame
   edge, and it's a different league (NAO robots, different camera).
   The NAO robots' glowing blue LED eyes also cause false alarms there.
3. **Blurry or partly hidden balls** are missed ~25% of the time (vs 8%
   for clear ones) — TORSO-21's `blurred` / `concealed` flags. The robot's
   camera is on a walking head, so motion blur is normal in real games.
4. **Unlabeled balls.** Several "false alarms" (especially in the
   "Sydney" recordings) are real balls in the background or at someone's
   feet that TORSO-21 has no box for. Worth a label-checking pass on
   anything we use for testing.
5. **Double boxes on very close balls.** When the ball fills much of the
   frame, the model sometimes draws two overlapping boxes; one counts as
   a false alarm.

## Training on Google Colab

A laptop gets very hot training for hours. `notebooks/train_on_colab.ipynb`
runs the whole pipeline on Google Colab's free GPU instead: open it from
GitHub in Colab, choose *Runtime → Change runtime type → GPU*, and *Run
all*. It clones this repo, downloads TORSO-21, builds the dataset the
config needs (data-team's `split.py`, plus `augment.py` for
`ball_v2_aug`), trains the config you pick (default
`configs/ball_v2.yaml`), evaluates it on the test split, and copies the
model, its `.json` and a visual report to your Google Drive. If Colab
disconnects, re-run it — it resumes from the last saved epoch on your
Drive.

## First task

Run a pretrained YOLO model (via `ultralytics`) on a sample image or
webcam frame and confirm the environment works end to end — that's the
Sept baseline everything else this semester builds on.
