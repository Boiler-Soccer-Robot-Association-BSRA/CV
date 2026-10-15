# TORSO-21 exploration notes (Sept Week 2)

Background reading for data-team. The concrete file format lives in the
["Dataset schema"](../README.md#dataset-schema) section of the
data-team README; this page covers the *why*: what's in TORSO-21, how
RoboCup images differ from the data stock detectors were trained on, and
what's missing for our five classes.

## 1. What TORSO-21 is

**TORSO-21** ("Typical Objects in RoboCup Soccer 2021") was published by
the Hamburg Bit-Bots at the RoboCup 2021 Symposium. It was built so teams
in the Humanoid League (HSL) could compare vision approaches on a common
benchmark instead of each using their own lab images.

- **Real world — 10,464 images.** Collected from RoboCup competitions,
  the Bit-Bots' lab, and images other teams uploaded to the ImageTagger
  labeling platform. Mostly HSL KidSize robots, plus some Standard
  Platform League (SPL, NAO robots) images. Many different cameras and
  lenses, so resolutions vary between imagesets.
- **Simulation — 24,000 images.** Rendered in Webots on the official
  KidSize field, with six robot models and randomized camera height and
  field of view. Comes with perfect labels, depth images and 6D poses.
- **Diversity was engineered on purpose.** Robot video is extremely
  repetitive (30 near-identical frames per second), so the authors ran
  ~44k candidate images through a variational autoencoder and greedily
  dropped near-duplicates. They also excluded sets where the camera was
  static, the ball was always centered, or no robots were moving.
  Takeaway for us: *don't* split TORSO-21 by random frame — images from
  the same imageset are still correlated, so split by `imageset_id`
  (see Week 4).
- **Labels:** ball and robot boxes, goalpost quadrilaterals, L/T/X line
  intersections as points, and per-pixel field + line segmentation.
  Robots also have team color (blue/red/unknown) and player number.
  Objects can be flagged `blurred` or `concealed`, and absent classes are
  explicitly marked `in_image: false`.

Other RoboCup datasets worth knowing about (not downloaded yet):

| Dataset | What it adds |
|---|---|
| NUbots [YOLOSoccer](https://huggingface.co/datasets/NUbots/YOLOSoccer) / [segmentation](https://huggingface.co/datasets/NUbots/segmentation) (Hugging Face) | TORSO-21 already converted to YOLO / segmentation, plus NUbots' own lab and synthetic (NUpbr) images. Handy cross-check for our converter. |
| rUNSWift SPL Field Segmentation Dataset | Field-region segmentation from NAO cameras (SPL). |
| SoS: Semi-Synthetic Soccer dataset (RoboCup 2024) | Synthetic + real composited images for segmentation. |
| RoboCupSimData | 2D/3D *simulation league* logs — positions, not images. Not useful for detection. |

## 2. How RoboCup imagery differs from general-purpose datasets

Stock detectors (our `yolov8n.pt`) are trained on COCO: photos taken by
people at eye level of 80 everyday classes. RoboCup images break most of
COCO's assumptions:

- **Color is dominated by one green.** Most pixels are artificial turf.
  Ball, lines and goalposts are all mostly **white**, so the model must
  rely on shape and context, not color. White goalposts look like
  vertical field lines; a white ball on a white line nearly disappears.
- **Lighting is uncontrolled.** Competition halls mix ceiling lights and
  daylight from windows, giving bright spots, hard shadows from robots,
  color casts, and over/under-exposed frames, and conditions change
  between venues. This is why Week 5 augmentation focuses on brightness,
  contrast and color jitter.
- **Camera viewpoint.** The camera is in a ~40-90 cm robot's head,
  usually tilted down at the ground — the "horizon" can be anywhere or
  absent. Objects close to the robot are huge and cut off; far ones are a
  few pixels. Wide-angle lenses add barrel distortion (integration-team
  calibrates this out later, but training images are distorted).
- **Motion blur and shake.** Walking robots bob and sway, so many frames
  are blurred (TORSO-21 flags these with `blurred: true`).
- **Small objects.** A ball across the field can be ~10 px wide. Small-
  object recall matters more than in COCO, which affects input
  resolution choices for detection-team.
- **Heavy occlusion.** Robots crowd around the ball, so balls are often
  partly hidden behind legs (`concealed: true`).
- **Off-field clutter.** Above the field there are spectators, banners,
  other fields and white objects that look like balls. TORSO-21's field
  segmentation lets us ignore things above the field boundary.
- **Robots don't look like a class.** Every HSL team builds its own robot,
  so "robot" varies wildly in shape and color; COCO's closest class is
  `person`. In SPL every robot is a NAO, which is the opposite problem.
- **Class meaning.** COCO's `sports ball` covers any ball; RoboCup needs a
  specific (mostly white, FIFA-pattern) ball and adds classes COCO
  doesn't have at all: goalposts, field lines, line intersections.

## 3. How bounding-box labels are represented

The same box can be written four common ways — mixing them up is the
most common source of silently wrong labels:

| Format | Box | Units |
|---|---|---|
| TORSO-21 | `[[x_min, y_min], [x_max, y_max]]` | pixels, origin top-left |
| Pascal VOC | `xmin ymin xmax ymax` | pixels |
| COCO | `[x_min, y_min, width, height]` | pixels |
| YOLO (what we train with) | `class cx cy w h` | fractions of image width/height |
| `shared.types.BoundingBox` | `x_min, y_min, x_max, y_max` | pixels |

Worked example — the ball `[[261, 215], [292, 245]]` in a 640 x 480 image
becomes YOLO `0 0.4320 0.4792 0.0484 0.0625` (class 0 = `ball`).

Because TORSO-21 mixes resolutions, always normalize with the
`width`/`height` of *that* image, never a hardcoded size.

TORSO-21 also uses two non-box geometries: **4-point polygons** for
goalposts (convert with min/max of the points) and **single points** for
line intersections (no size information at all).

## 4. Gaps against `shared/classes.py`

| Class | Status | What's missing / to decide |
|---|---|---|
| `ball` | Covered | Nothing. ~6k real labels — enough for the October milestone. |
| `robot` | Covered | Nothing for detection. Labels include team color/number if we want them later. |
| `goalpost` | Covered after conversion | Polygon → box. Real images only label posts on the goal line — **no crossbar / full-goal labels** (simulation has them). |
| `field_line` | **No box labels** | Lines exist only as segmentation pixels. Options: **(a)** treat `field_line` as a segmentation output (YOLO-seg or a small seg head) using TORSO-21 masks as-is — recommended; **(b)** cut the line mask into short segments and box each — doable with OpenCV (`connectedComponents` / Hough) but boxes around diagonal lines are mostly empty grass. Needs a decision with detection-team before November. |
| `landmark` | **Undefined + point-only** | 1. We need to define "landmark". If it means line intersections (L/T/X, which include penalty mark and center point), TORSO-21 has ~26k of them. 2. They're points, so we need a rule for box size (e.g. fixed fraction of image height, or scale with distance / y-position). 3. Does it need sub-types (L vs T vs X)? Localization usually wants them, which would mean splitting `landmark` into several classes. 4. Nothing for goal-area corners, center circle, or off-field markers. |

Other gaps worth flagging:

- **No validation split** — only train/test. Make val from train, split
  by `imageset_id` to avoid near-duplicate leakage.
- **No outdoor / real-grass images** — the authors list this as future
  work. If we ever play outdoors, we need our own data.
- **Our own robot and venue are not in it.** Expect a domain gap on our
  camera; plan to label a few hundred of our own frames (Nov/Dec hard
  cases).
- **`obstacle` labels** exist in the files but aren't documented upstream.
  Ignored for now.
- **Segmentation mask colors** still need to be read off the real files
  with `scripts/explore_torso21.py` and recorded in the README.

## Sources

- TORSO-21 repository: https://github.com/bit-bots/TORSO-21-dataset
- Paper: Bestmann et al., *TORSO-21 Dataset: Typical Objects in RoboCup
  Soccer 2021*, RoboCup 2021: Robot World Cup XXIV, LNCS 13132.
  https://link.springer.com/chapter/10.1007/978-3-030-98682-7_6
- Sample annotations (NUbots mirror):
  https://huggingface.co/datasets/NUbots/segmentation
