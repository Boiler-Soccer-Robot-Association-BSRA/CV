# testing-deployment-team

**Members:** Hossein Hatami Yazd

Owns evaluation, robustness testing, benchmarking, optimization research,
and pulling the semester's work into one documented, reproducible result.
See [ARCHITECTURE.md](../ARCHITECTURE.md) for how this touches both
detection-team and integration-team's output.

## Semester roadmap

### September — Fundamentals & setup

- **Week 1** — Install the root [requirements.txt](../requirements.txt);
  set up the environment you'll use to run other teams' models.
- **Week 2** — Learn how bounding boxes and confidence scores work, and
  the definitions of precision, recall, and mAP — you'll be computing
  these for real starting in October. Run an existing pretrained
  detector yourself and manually inspect its output.
- **Week 3** — Sketch the evaluation harness's interface (what it takes
  in, what it reports) so it's ready to point at detection-team's first
  model. *End-of-September milestone (team-wide): everyone can
  load/manipulate images with OpenCV and run an existing detector.*

### October — First RoboCup detection system

- **Week 4** — Build the evaluation harness (`tests/eval_harness.py`):
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
- **Week 8** — Help validate ground-plane projection accuracy: compare
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

1. Computes precision and recall at IoU 0.5, plus mean ms/frame.
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
`detection-team/inference/detector.py`'s `Detector` already has, so no
adapter code is needed once that's implemented for real.

Once a real model exists, re-run the harness against a real held-out
set and overwrite `benchmarks/baseline.json` with those numbers (commit
that change on its own so it's easy to see when/why the baseline
moved).

## First task

Sketch the evaluation harness: given a trained model and a val split, run
inference and report precision/recall/mAP — mirroring what detection-team
needs for its Oct milestone. Coordinate with detection-team so you're not
building two versions of the same thing.
