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
├── scripts/      # download.py, explore_torso21.py, convert.py, split.py, augment.py
├── labeling/     # labeling tool config/output
├── docs/         # exploration notes (e.g. torso21-exploration.md)
└── README.md
```

## Dataset schema

Source: [TORSO-21](https://github.com/bit-bots/TORSO-21-dataset)
(Bestmann et al., RoboCup 2021, MIT license). Background, how RoboCup
imagery differs from COCO-style data, and the full gap analysis are in
[docs/torso21-exploration.md](docs/torso21-exploration.md).

> **Verify after downloading.** This section was written from the
> upstream README, the paper, and sample annotation entries. Once you have
> the data, run `scripts/explore_torso21.py` on each split and correct
> anything below that doesn't match (especially resolutions and mask
> colors).

### Getting it

```bash
python data-team/scripts/download.py --subset reality --split test   # small first look
python data-team/scripts/download.py --subset reality                 # real-world train + test
python data-team/scripts/explore_torso21.py data-team/datasets/TORSO-21/reality/test --viz 12
```

Everything lands in `data-team/datasets/TORSO-21/` (gitignored). The
simulation subset (24,000 Webots images) is large and not needed for
the ball milestone; fetch it with `--subset simulation` only if we decide
to use it.

### Folder layout

```
TORSO-21/
├── reality/               # 10,464 real images (HSL KidSize + SPL)
│   ├── train/
│   │   ├── annotations.yaml
│   │   ├── images/        # file names start with the imageset id, e.g. 1028-img00691.png
│   │   └── segmentations/ # one mask PNG per image, same stem
│   └── test/              # same structure
└── simulation/            # 24,000 Webots images (85% train / 15% test)
    ├── train/             # same as above, plus depth/
    └── test/
```

There is **no validation split** — we carve one out of `train/` in
`scripts/split.py` (Week 4). Keep `test/` untouched as our held-out set.

### `annotations.yaml`

One YAML file per split. Top-level key `images`, keyed by file name:

```yaml
images:
  1028-img00691.png:
    id: 688942
    width: 640               # resolutions vary between imagesets —
    height: 480              # always read these per image
    metadata:
      League: HSL            # HSL or SPL
      location: "..."        # event / lab where it was recorded
      fov: 42                # may be null
      imageset_id: 130
      tags: [...]
    annotations:
      - type: ball
        in_image: true
        blurred: false
        concealed: false
        vector: [[261, 215], [292, 245]]
      - type: robot
        in_image: true
        blurred: true
        concealed: false
        color: blue          # blue | red | unknown   (robots only)
        number: 4            # 1-6 or null            (robots only)
        vector: [[251, 123], [328, 239]]
      - type: goalpost
        in_image: true
        blurred: false
        concealed: false
        vector: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
      - type: T-Intersection
        in_image: true
        blurred: false
        concealed: false
        vector: [[237, 213]]
      - type: X-Intersection
        in_image: false      # "checked, none in this image" — no vector
```

Field conventions:

| Field | Meaning |
|---|---|
| `type` | Object class (see table below). Case-sensitive: `L-Intersection`, not `l_intersection`. |
| `in_image` | `false` = an annotator confirmed this class is absent from the image (a negative label). These entries have no `vector`. Skip them when building boxes, but they tell you the image was fully labeled. |
| `blurred` | Object is motion-blurred (common: the camera is on a walking robot's head). |
| `concealed` | Object is partially hidden (occluded by another object, or cut off at the image edge). |
| `vector` | List of `[x, y]` **pixel** coordinates, origin at the **top-left** of the image, x to the right, y down. Not normalized. |

### Classes and geometry

| TORSO-21 `type` | Geometry (`vector`) | Real-world count* | Notes |
|---|---|---|---|
| `ball` | 2 points: `[[x_min, y_min], [x_max, y_max]]` — axis-aligned box | 6,081 | |
| `robot` | 2 points — axis-aligned box | 7,641 | Has `color` and `number` attributes |
| `goalpost` | 4 points — quadrilateral around the post | 7,888 | Only posts on the goal line; no crossbar in real images |
| `L-Intersection` | 1 point — center of the line corner | 10,375 | |
| `T-Intersection` | 1 point | 8,659 | |
| `X-Intersection` | 1 point | 7,268 | Also used for the penalty mark and center point |
| `obstacle` | 2 points — box | not reported | Appears in the files but not in the upstream README; ignore for now |
| *(segmentation)* | Mask PNG in `segmentations/` | 10,464 | 3 classes: background, field, lines |

\* Object counts from the upstream README for `reality/` (train + test).

**Segmentation masks** are color PNGs, one per image, with the same file
stem as the image. Each pixel is background, field, or line. Run
`explore_torso21.py` to print the exact RGB value of each class and
record them here: background = `TODO`, field = `TODO`, lines = `TODO`.
(Upstream replaced the real-world masks on 2021-07-12 because of wrong
field color values, so make sure you have the current download.)

**Simulation only:** each object also carries a 6D `pose`, each image a
`camera_pose`, the full goal (including crossbar) is labeled, and a
`depth/` image is provided.

### Mapping to our classes (`shared/classes.py`)

| Our class | From TORSO-21 | Ready to use? |
|---|---|---|
| `ball` | `ball` box | **Yes** — direct. |
| `robot` | `robot` box | **Yes** — direct. Drop `color`/`number` for now. |
| `goalpost` | `goalpost` 4-point polygon → axis-aligned box (min/max of the 4 points) | **Yes**, after conversion. |
| `field_line` | Line pixels in the segmentation masks | **No boxes exist.** Lines are long, thin and diagonal, so a box is a poor fit. Needs a decision with detection-team (see notes). |
| `landmark` | `L-`/`T-`/`X-Intersection` points | **Partially.** Points only — no box size. TORSO-21 has no goal-box corners, center circle, or off-field markers. Needs a definition + a box-size rule. |

The same mapping is coded as `TORSO_TO_OURS` in
`scripts/explore_torso21.py`; `scripts/convert.py` should import it
rather than redefining it.

### Converting to YOLO (for `scripts/convert.py`)

Our pipeline uses YOLO labels: one `.txt` per image, one line per object,
`<class_id> <cx> <cy> <w> <h>`, all normalized to 0-1 by **that image's**
width and height. `class_id` comes from `shared.classes.CLASS_TO_ID`.

```
x_min, y_min = min of the vector's x / y      x_max, y_max = max of the vector's x / y
clip to [0, width] / [0, height]
cx = (x_min + x_max) / 2 / width              w = (x_max - x_min) / width
cy = (y_min + y_max) / 2 / height             h = (y_max - y_min) / height
```

Rules: skip `in_image: false`; skip boxes with zero width or height after
clipping; keep `blurred` and `concealed` objects in training (they're the
hard cases we want), but report metrics with and without them. Images
with no remaining objects still get an empty `.txt` (a negative example).

Coordinate formats you'll run into, for reference:

| Format | Box representation | Units |
|---|---|---|
| TORSO-21 | `[[x_min, y_min], [x_max, y_max]]` | pixels |
| Our `shared.types.BoundingBox` | `x_min, y_min, x_max, y_max` | pixels |
| Pascal VOC (XML) | `xmin, ymin, xmax, ymax` | pixels |
| COCO (JSON) | `[x_min, y_min, width, height]` | pixels |
| YOLO (txt) | `class cx cy w h` | normalized 0-1 |

## Setup

Uses the root [requirements.txt](../requirements.txt). No additional
dependencies yet — if a labeling tool or augmentation library needs one,
add it here and note it in this README.

## First task

Download the TORSO-21 dataset (see `scripts/download.py`) and document
its annotation format and class list in the "Dataset schema" section
above — that's the Sept milestone and what detection-team is waiting on.
