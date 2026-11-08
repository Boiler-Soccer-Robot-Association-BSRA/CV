# Splits and augmentation (Oct Weeks 4-5)

How we turn TORSO-21 into the train/val/test sets detection-team trains
on, and how we augment the training set. Scripts: `scripts/split.py`,
`scripts/augment.py`. Tests: `data-team/tests/` (run `pytest data-team/tests`).

## Quick start

```bash
# 1. Download (once, ~5.7 GB) - see "Dataset schema" in data-team/README.md
python data-team/scripts/download.py --subset reality

# 2. Ball split, YOLO format, seed 42 -> data-team/datasets/ball/
python data-team/scripts/split.py

# 3. One augmented copy of every training image + 12 before/after previews
python data-team/scripts/augment.py data-team/datasets/ball --preview 12
```

The output follows the **Dataset contract** in detection-team's README,
so detection-team only has to point their config at
`data-team/datasets/ball/data.yaml` (or `data_aug.yaml` for the
augmented version) instead of their placeholder.

## The seed

**Team seed: `42`** (`DEFAULT_SEED` in `split.py`; `augment.py` uses the
same default). Same TORSO-21 download + same seed = exactly the same split
on everyone's computer. To check, compare the `fingerprint` in
`split_info.yaml` - same fingerprint means identical splits.

Don't change the seed without telling the team: model scores are only
comparable on the same split.

## How the split works

- **TORSO-21's `test/` is not our test set.** Every recording in its
  `test/` folder also appears in `train/`, so it isn't unseen footage.
  `split.py` pools `train/` and `test/` and splits the pile itself.
- **Split by recording, never by image.** The number at the start of a
  file name (e.g. `1028` in `1028-img00155.png`) is the recording. Frames
  of one recording are near-duplicates: if the model trained on frame 10
  and was tested on frame 11, scores would look much better than reality.
  Every frame of a recording goes to the same split, and the script
  refuses to write a split where a recording shows up twice.
- **Sizes: ~80 / 10 / 10 by ball count** (`--val 0.1 --test 0.1`).
  Recordings are shuffled with the seed, then put into test and val
  until each holds ~10% of all ball boxes; the rest is train. Two guards:
  a recording with more than 25% of the val/test target never goes there,
  and one that would push a split more than 15% past its target is
  skipped (both go to train). TORSO-21 has a few huge recordings (up to
  ~1000 balls); without the guards, two of them made up the entire test
  set. With them (seed 42): val has 18 recordings, test has 8. Recordings
  are always assigned by their *ball* count, whatever `--classes` is, so
  the ball dataset and the multi-class dataset test on exactly the same
  recordings and their ball scores can be compared. Recordings vary in
  size, so the real shares are approximate - the script prints them.
- **Images kept:** every image with an object of the chosen classes, plus
  "negatives" equal to 10% of those (`--negatives 0.1`): images where
  annotators *confirmed* none of the chosen classes is present. They teach
  the model not to fire on lines, robot feet and goalposts. Images where
  the class wasn't labeled either way are never used as negatives.
- **Class ids follow `shared/classes.py`** (0 ball, 1 robot, 2 goalpost,
  3 field_line, 4 landmark); TORSO-21 types are mapped with
  `TORSO_TO_OURS` from `explore_torso21.py`. `data.yaml` names every id
  from 0 up to the highest one used (YOLO needs them contiguous).
- **landmark** = all L, T and X intersections. They're single points, so
  each gets a square box 5% of the image width wide, centered on the
  point (`--landmark-size 0.05`).
- **field_line is not produced** - no boxes from line masks (team
  decision; lines will be handled another way later).
- **More classes:** `python data-team/scripts/split.py --classes ball robot goalpost landmark`
  writes `data-team/datasets/multiclass/`.
- Images are **symlinked**, not copied. Use `--copy` to move the folder
  to another computer.

### Forcing recordings into test: `--move-to-test`

```bash
python data-team/scripts/split.py --move-to-test 180 --out data-team/datasets/ball_r180
```

`--move-to-test` takes one or more recording numbers and puts every frame
of those recordings into **test**, wherever the seeded split would have
placed them. Every other recording stays where it was, so the moved
recordings simply leave train (or val).
An unknown recording number stops the script with an error. The moved
recordings are listed under `moved_to_test` in `split_info.yaml`.

Use it when the test set is missing a kind of footage. Recording 180
(Iran Open 2018, Humanoid League, robot camera) is moved to test because
the default test set's tiny balls (< 20 px wide) were almost all one
stationary ball in recording 151, repeated in 113 frames - so test wasn't
really measuring tiny balls. With 180 moved, test has 871 balls
(seed 42).

Notes:

- Use a **new `--out` folder** (like `ball_r180`) so the default
  `datasets/ball/` split, and models' results on it, stay as they were.
- Moving recordings changes train, so a model trained on `ball/` must not
  be evaluated on `ball_r180/`'s test set: recording 180 was in its
  training data. Retrain on the new split first.
- Moved recordings don't count toward the ~10% targets, so test ends up
  bigger than 10%.
- Review of the tiny balls in the `ball_r180` test set (Nov 2026): 6 of
  the 63 "tiny" boxes outside recording 151 are big balls cut off at the
  image edge (narrow only because they're clipped), and 4 in recording 97
  have no visible ball (likely hidden behind a robot's foot, but not
  marked `concealed`). No other recording has a ball repeated in the
  same spot. The 4 hidden balls are now dropped by `label_fixes.yaml`
  (next section); the clipped balls are real and stay.

### Dropping wrong labels: `label_fixes.yaml`

`split.py` drops every box listed in
[`data-team/label_fixes.yaml`](../label_fixes.yaml): TORSO-21 labels with
nothing visible in them, which no model could find. Each entry names the
image, the class, the box in TORSO-21's pixels and the reason.

- The only rule for adding one: **nothing visible in the box**. Write the
  reason, so every model is scored the same way.
- A fix never moves a recording: recordings are still assigned to splits
  by their original ball counts. An image left with no boxes is dropped
  like any other unlabeled image.
- A fix whose image is in the download but whose box isn't stops the
  script, so a stale file can't go unnoticed.
- `split_info.yaml` records `label_fixes_applied` (boxes dropped).
  `--no-label-fixes` builds the split without them; `--label-fixes PATH`
  uses another file.

The 4 fixes so far are all in recording 97's test frames, so training
data is unchanged and models trained before them don't need retraining:
just re-run `split.py` and evaluate again. With them, the default split's
test set has 620 balls (fingerprint `5e958505b9bbcd71`) and
`ball_r180`'s has 867 (`963272dac0b46f47`).

Output:

```
data-team/datasets/ball/
├── data.yaml / data_aug.yaml
├── split_info.yaml      # seed, settings, moved_to_test, label_fixes_applied, recordings per split, counts, fingerprint
├── augment_info.json
├── images/{train,val,test,train_aug}/
├── labels/{train,val,test,train_aug}/
└── aug_preview/         # with --preview
```

## Augmentation

Only the **train** split is augmented (the script refuses `val`/`test`).
Each augmented copy gets a random mix of:

| Augmentation | What it does | Why (RoboCup) | Default chance |
|---|---|---|---|
| Lighting | Contrast x0.6-1.4, brightness ±40, gamma 0.6-1.6, warm/cool color cast ±12% | Halls mix ceiling lights and daylight; every venue is different | 80% |
| Blur | Gaussian (kernel 3-7) or motion blur (length 5-15 px, any angle) | Camera on a walking, shaking robot head | 30% |
| Crop/scale | Zoom in to 60-100% of the image, or zoom out to ~77-100% with gray padding | Ball appears at many sizes and positions | 50% |

Boxes are moved with the crop/scale. A box is dropped if less than 40% of
it is still visible; the script retries a few times so a ball image
doesn't turn into an empty one. Every copy gets at least one augmentation.

Options: `--copies N` (augmented copies per image), `--p-light`,
`--p-blur`, `--p-crop` (chances), `--seed`.

**Always look at the previews** (`ball/aug_preview/`, original on the
left, augmented on the right, boxes drawn) after changing any setting - if
the boxes don't sit on the balls, something is wrong.

Note: Ultralytics already applies its own random augmentation while
training (color jitter, flips, mosaic). Ours adds the RoboCup-specific
cases (blur, strong lighting changes) and makes them visible and
reproducible. Week 6 error analysis will tell us whether it helps:
train once with `data.yaml` and once with `data_aug.yaml` and compare.

## Open questions for the team

- Confirm seed 42 (detection-team's placeholder used seed 0 and a 75/10/15
  split, so ball_v1's numbers aren't directly comparable with models
  trained on this split).
- Is 10% val / 10% test enough? Splitting by recording makes the sizes
  lumpy (a few big recordings); check the shares `split.py` prints.
- How many augmented copies? Start with 1, compare against no
  augmentation, then decide.
