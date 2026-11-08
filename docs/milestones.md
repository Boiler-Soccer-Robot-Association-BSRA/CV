# Milestones

Per-module owner, measurable target, and target date, so "done" means
something concrete instead of "it runs."

**This is a draft.** The rows below are seeded from the semester
timeline already in [README.md](../README.md#semester-timeline) and
each subteam's roadmap — the targets are illustrative starting points,
not commitments anyone has actually signed off on yet. Each subteam
should confirm or adjust its own rows (owner, exact metric, date) and
open a PR updating this file once they have.

Last reviewed: **Nov 8, 2026**.

Update this table whenever a milestone is hit, missed, or rescheduled —
a stale milestones doc is worse than none, same rule as the
[Pipeline status](../ARCHITECTURE.md#pipeline-status) table.

| Module | Owner(s) | Target metric | Target date | Status |
|---|---|---|---|---|
| Dev environment & tooling | Whole team | All 7 members run `./scripts/setup.sh` and `python scripts/quickstart.py` successfully | Sept 30 | **Done** — `setup.sh` and `quickstart.py` exist and CI installs the environment; closed by the CV lead Nov 8 |
| Ball detection (detection-team) | Suhaas, Aditya Mitra | 90% recall @ IoU 0.5 on a held-out RoboCup validation split, ≥30 FPS on a MacBook (CPU) | Nov 15 (moved from Oct 31) | **Done (borderline)** — team decision Nov 8: counts as met, and both recall numbers are always reported. `ball_v4` on the test set with recording 180 held out (867 balls after dropping 4 labels with no visible ball, see `data-team/label_fixes.yaml`): **0.915 recall** counting recording 151's stationary ball once (**0.803 as scored**), 0.921 precision at conf 0.20, 31 FPS on CPU (43 FPS as ONNX). Tiny balls are the weak spot: 42 of 58 (72%) found outside recording 151. `ball_v2`, `ball_v3_960` and `ball_v2_aug` didn't beat it. See [detection results](../detection-team/README.md#results) |
| Evaluation & benchmark harness (testing-deployment-team) | Hossein, Henry | `eval_harness.py` runs in CI on every PR and gates on `benchmarks/baseline.json`; baseline reflects a real trained model, not just the dummy detector | Oct 15 | **Overdue, unblocked** — harness done (precision, recall, mAP@0.5, ms/frame; runs in CI on every PR against the dummy detector). The two blockers are gone: data-team's held-out test split and a trained model (`ball_v4`) both exist. Next: set `benchmarks/baseline.json` from `ball_v4` on the test split |
| Multi-class detection (detection-team) | Suhaas, Aditya Mitra | ≥80% recall @ IoU 0.5 across ball, robot, goalpost and landmark (field_line was taken out of the box detector — team decision, Oct 15) | Nov 30 | **At risk** — `multiclass_v1` (placeholder split): 0.786 recall overall, AP 0.86 ball / 0.82 goalpost / 0.76 robot / 0.73 landmark. `multiclass_v2` on data-team's split (with recording 180 held out) is the real check and hasn't been trained yet; Colab GPU quota is the bottleneck |
| Ground-plane projection (integration-team) | Phil, Henry | Robot-relative position error ≤ 0.3 m at 2 m range, measured against a known checkerboard/reference-point layout | Nov 30 | **At risk** — `calibration/calibrate.py` is done and tested on rendered board images; `geometry/projection.py` is still a skeleton. The camera (e-con See3CAM_24CUG) is ordered but hasn't arrived, so no real calibration yet. Projection can be built and tested on synthetic data in the meantime |
| ROS 2 perception integration (integration-team) | Phil, Henry | Perception messages published at ≥10 Hz with the schema documented in [ARCHITECTURE.md](../ARCHITECTURE.md#data-contracts) | Dec 15 | **At risk** — the message schema is defined (four `.msg` files, #33, matching `shared/types.py`); nothing publishes them yet, and only the onboarding publisher/subscriber nodes exist |
| Final documentation & demo | Whole team | A new member can reproduce the pipeline end to end from README + ONBOARDING alone | Dec 15 | **In progress** — setup, data, training and evaluation are documented; the end-to-end part depends on projection and ROS 2 |

## Template row

Copy this when adding a milestone:

```
| <module> | <owner(s)> | <measurable target, with a number> | <date> | <Not started / In progress / Done> |
```
