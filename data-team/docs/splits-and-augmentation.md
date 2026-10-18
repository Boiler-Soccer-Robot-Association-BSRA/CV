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

Output:

```
data-team/datasets/ball/
├── data.yaml / data_aug.yaml
├── split_info.yaml      # seed, settings, recordings per split, counts, fingerprint
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
