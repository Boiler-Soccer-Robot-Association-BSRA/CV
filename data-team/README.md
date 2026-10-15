# data-team

**Members:** Gabriella, Josephine

Owns dataset acquisition, labeling, splits, and augmentation for the
perception pipeline. See [ARCHITECTURE.md](../ARCHITECTURE.md) for how
this feeds into detection-team.

## Semester roadmap

### September — Fundamentals & setup

- **Week 1** — Environment setup: Python/Git installed, repo cloned, root
  `requirements.txt` installed. Agree on where the team will store raw
  datasets (a drive folder — see Team norms in the root README).
- **Week 2** — Learn the domain: what TORSO-21 and other RoboCup datasets
  contain, how RoboCup imagery differs from general-purpose datasets
  (field colors, lighting, camera angles), and how bounding-box labels
  are represented. Download and explore the
  [TORSO-21](https://github.com/bit-bots/TORSO-21-dataset) dataset,
  catalog its classes and annotation format, and note what's missing
  for the classes we ultimately need (`shared/classes.py`: ball, robot,
  goalpost, field_line, landmark).
- **Week 3** — Document the dataset schema in the "Dataset schema"
  section below. *End-of-September milestone (team-wide): everyone can
  load/manipulate images with OpenCV and run an existing detector.*

### October — First RoboCup detection system

- **Week 4** — Build the train/val/test split script (`scripts/split.py`)
  against the ball subset of the dataset; agree on a fixed random seed
  so splits are reproducible.
- **Week 5** — Implement data augmentation (`scripts/augment.py`):
  lighting changes, blur, crop/scale.
- **Week 6** — Work with detection-team and testing-deployment-team on
  error analysis from the first trained model; expand or re-augment the
  ball dataset to cover what it's getting wrong; finalize dataset
  documentation and confirm the split pipeline is reproducible end to
  end. *End-of-October milestone (team-wide): a reproducible ball
  detector on unseen footage.*

### November — Perception beyond bounding boxes

- **Week 7** — Begin extending the dataset with robot and goalpost
  labels.
- **Week 8** — Extend to field-line and landmark labels; hand off
  updated splits to detection-team as they become ready.
- **Week 9** — Curate harder validation sets — varied lighting, partial
  occlusion, varied backgrounds — for testing-deployment-team's Dec
  robustness testing, and finalize the expanded multi-class dataset.
  *End-of-November milestone (team-wide): a prototype pipeline detects
  the ball and converts it to an approximate robot-relative position.*

### December — Documentation & integration

- **Week 10** — Support detection-team and testing-deployment-team with
  additional hard-case data (specific lighting/angle/occlusion examples
  they need for robustness testing).
- **Week 11** — Final dataset documentation pass: write up the full
  dataset workflow (download → convert → split → augment) in this
  README so a new member could reproduce it from scratch.
- **Week 12** — Support final perception demo prep as needed.
  *End-of-semester milestone (team-wide): a working, documented RoboCup
  perception prototype.*

## Folder layout

```
data-team/
├── datasets/     # raw and processed datasets (gitignored, do not commit)
├── scripts/      # download.py, convert.py, split.py, augment.py
├── labeling/     # labeling tool config/output
└── README.md
```

## Dataset schema

_To be filled in once the TORSO-21 dataset is downloaded and explored —
document the label format (annotation file structure, class list,
coordinate conventions) here so detection-team and testing-deployment-team
can rely on it._

## Setup

Uses the root [requirements.txt](../requirements.txt). No additional
dependencies yet — if a labeling tool or augmentation library needs one,
add it here and note it in this README.

## First task

Download the TORSO-21 dataset (see `scripts/download.py`) and document
its annotation format and class list in the "Dataset schema" section
above — that's the Sept milestone and what detection-team is waiting on.
