# BSRA-CV

Boiler Soccer Robots Association — Computer Vision team

**New to the team? Start with [ONBOARDING.md](ONBOARDING.md)** — it
tells you what to read, in what order, and how to get set up.

## Quickstart

```bash
git clone https://github.com/hhatamiy/BSRA-CV.git && cd BSRA-CV
./scripts/setup.sh && source venv/bin/activate   # one-command env setup
pytest testing-deployment-team/tests/            # unit tests
ruff check .                                     # lint
python testing-deployment-team/tests/eval_harness.py   # benchmark harness
```

See [Getting set up](#getting-set-up) below for the step-by-step version
and what each command does.

## Leadership

- **Computer Vision Lead:** Hossein Hatami Yazd ([hhatamiy@gmail.com](mailto:hhatamiy@gmail.com), [hhatamiy@purdue.edu](mailto:hhatamiy@purdue.edu))
- **Programming Director:** Adhitya Vasudevan ([adhiv2007@gmail.com](mailto:adhiv2007@gmail.com), [vasude29@purdue.edu](mailto:vasude29@purdue.edu))

## What we're building

Vision code for the SUSTAINA-OP2 humanoid platform
([SUSTAINA-OP2](https://github.com/SUSTAINA-OP2)). The semester goal is
RoboCup-style object detection — ball, robots, goalposts, field
lines/landmarks — with a reproducible training pipeline, feeding into
robot-relative position estimates that the rest of the robot stack can
use. We're starting from a pretrained YOLO model and fine-tuning it,
rather than building detection from scratch.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pipeline fits
together and where each subteam's code plugs in.

## Subteams

Six people across four subteams. Each folder has its own README with
that team's semester roadmap, folder layout, and first task. See
[.github/CODEOWNERS](.github/CODEOWNERS) for who's on each one.

- [data-team/](data-team/) — dataset acquisition, labeling, splits, augmentation (Gabriela, Josephine)
- [detection-team/](detection-team/) — model training and the core detector (Suhaas, Aditya Mitra)
- [integration-team/](integration-team/) — calibration, geometry, ROS 2 packaging (Phil)
- [testing-deployment-team/](testing-deployment-team/) — evaluation, robustness, benchmarking, final packaging (Hossein)

Code shared across all four lives in [shared/](shared/) (class name
constants, config loading, common types).

## Getting set up

You'll need Python 3.10 (pinned in [`.python-version`](.python-version);
tested on Apple Silicon Macs and Linux). Everyone should work inside a
virtual environment so dependencies stay consistent across machines —
`requirements.txt`/`requirements-dev.txt` pin exact versions for that
reason, so `pip install` gives everyone the same environment.

One command:

```bash
git clone https://github.com/hhatamiy/BSRA-CV.git
cd BSRA-CV
./scripts/setup.sh          # creates venv/, installs deps, installs pre-commit hooks
source venv/bin/activate    # do this in every new shell
python scripts/quickstart.py
```

Or by hand, if you want to see each step:

```bash
# clone the repo
git clone https://github.com/hhatamiy/BSRA-CV.git
cd BSRA-CV

# create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# install shared + dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install

# run the quickstart to confirm your environment works
python scripts/quickstart.py
```

### Running tests and lint

```bash
pytest testing-deployment-team/tests/     # unit tests
ruff check .                              # lint
ruff format .                             # auto-format
```

### Running the benchmark harness

```bash
python testing-deployment-team/tests/eval_harness.py
```

Runs the current detector (the bundled dummy detector by default) over
the sample dataset in `testing-deployment-team/tests/fixtures/`, reports
precision/recall at IoU 0.5 and ms/frame, and fails (nonzero exit) if
recall has dropped too far versus
[`testing-deployment-team/benchmarks/baseline.json`](testing-deployment-team/benchmarks/baseline.json).
See [`testing-deployment-team/README.md`](testing-deployment-team/README.md#benchmark-harness)
for how to point it at a real model and dataset.

`scripts/quickstart.py` runs a sample image through as much of the real
pipeline as currently exists — see its docstring for exactly what that
covers today (honestly, not much yet — see the "Pipeline status" table
in [ARCHITECTURE.md](ARCHITECTURE.md#pipeline-status)) and what it
doesn't. The first run downloads ~6MB of pretrained YOLO weights.

The root `requirements.txt` (ultralytics, opencv-python, torch, numpy,
pyyaml) covers everyone. If your subteam needs something extra (ROS 2
packages, ONNX/TensorRT), add a `requirements.txt` inside that subteam's
folder rather than the shared one — see that team's README.

If you add a new shared dependency, pin it to an exact version in
`requirements.txt` (e.g. `some-package==1.2.3`) rather than leaving it
unpinned, so everyone installs the exact same setup.

## Semester timeline

By the end of Fall 2026 the CV team should be able to take RoboCup-style
camera footage, reliably detect important soccer-field objects, begin
converting those detections into robot-relative information, and have a
reproducible training/inference pipeline ready to integrate into the
full robot software stack.

- **Sept — Fundamentals & setup.** Every member can run the dev
  environment: OpenCV/PyTorch/Ultralytics installed, TORSO-21 explored,
  a pretrained detector running.
- **Oct — First RoboCup detection system.** A reproducible model that
  detects the soccer ball in unseen RoboCup-style footage.
- **Nov — Perception beyond bounding boxes.** Detection expands to
  robots/goalposts/field lines/landmarks; a prototype pipeline converts
  a ball detection into an approximate robot-relative position.
- **Dec — Documentation & integration.** Robustness testing, inference
  benchmarking, ONNX/TensorRT exploration, ROS 2 packaging, and enough
  documentation that a new member could reproduce the whole system.

Each subteam README has the full week-by-week breakdown for its slice
of this timeline: [data-team](data-team/README.md#semester-roadmap),
[detection-team](detection-team/README.md#semester-roadmap),
[integration-team](integration-team/README.md#semester-roadmap),
[testing-deployment-team](testing-deployment-team/README.md#semester-roadmap).
See [docs/milestones.md](docs/milestones.md) for the measurable,
owner-and-date version of these milestones.

### Success criteria

By the end of the semester, the team should ideally have:

- A reproducible CV development environment
- A shared RoboCup dataset workflow
- A trained RoboCup object-detection model
- Reliable ball detection
- Initial goalpost/robot/field perception
- Basic camera calibration
- Basic ground-plane projection
- Quantitative model evaluation
- Initial ROS 2 integration
- Clear documentation allowing a new member to reproduce the system
- A final perception demonstration

## Team norms

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full pull request
workflow (pre-commit hooks, what to run before opening a PR, review
requirements). Short version:

* Work off feature branches, open a pull request before merging into `main`.
* Keep large files (datasets, model weights) out of git. Use the `.gitignore` for that, and share large files through the [team drive folder](https://drive.google.com/drive/folders/1qa2ktrauYvNnrdFLqBTY9tBebvrS4sXG) instead.
* Post questions and blockers in the group chat between meetings rather than sitting stuck.
* Track physical equipment needs and purchase status in [EQUIPMENT.md](EQUIPMENT.md).

### Branching

Name feature branches `<subteam>/<short-task-name>` — lowercase,
hyphens instead of spaces, no ticket numbers needed for a club project:

* `data-team/torso21-download`
* `detection-team/baseline-yolo-train`
* `integration-team/camera-calibration`
* `testing-deployment-team/eval-harness`

`main` is protected: a pull request needs at least one approving review
and a passing CI check (`.github/workflows/lint-test.yml`) before it can
merge — budget time for that rather than expecting an instant merge.

## Meeting schedule

Weekly, Sundays 6–7 PM (subject to change).

**Open question:** now that the team is six people, worth deciding
whether to keep a single one-hour Sunday sync for everyone, or split
into a shorter full-team check-in plus separate subteam meetings.
Not decided yet — raise it at the next sync.

## Reference

* [SUSTAINA-OP2 platform](https://github.com/SUSTAINA-OP2)
* [Joseph Redmon: How computers learn to recognize objects instantly](https://www.ted.com/talks/joseph_redmon_how_computers_learn_to_recognize_objects_instantly)
