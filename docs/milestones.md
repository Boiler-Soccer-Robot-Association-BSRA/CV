# Milestones

Per-module owner, measurable target, and target date, so "done" means
something concrete instead of "it runs."

**This is a draft.** The rows below are seeded from the semester
timeline already in [README.md](../README.md#semester-timeline) and
each subteam's roadmap — the targets are illustrative starting points,
not commitments anyone has actually signed off on yet. Each subteam
should confirm or adjust its own rows (owner, exact metric, date) and
open a PR updating this file once they have.

Update this table whenever a milestone is hit, missed, or rescheduled —
a stale milestones doc is worse than none, same rule as the
[Pipeline status](../ARCHITECTURE.md#pipeline-status) table.

| Module | Owner(s) | Target metric | Target date | Status |
|---|---|---|---|---|
| Dev environment & tooling | Whole team | All 7 members run `./scripts/setup.sh` and `python scripts/quickstart.py` successfully | Sept 30 | **In progress** — `setup.sh` and `quickstart.py` exist and CI installs the environment; no member has confirmed running them yet |
| Ball detection (detection-team) | Suhaas, Aditya Mitra | 90% recall @ IoU 0.5 on a held-out RoboCup validation split, ≥30 FPS on a MacBook (CPU) | Nov 15 (moved from Oct 31) | **Borderline: met counting recording 151's stationary ball once, not as scored** — `ball_v4` on the harder test set (recording 180 added, 871 balls): **0.910 recall** counting that ball once (0.799 as scored), 0.921 precision at conf 0.20, 31 FPS on CPU (43 FPS as ONNX). Tiny balls are the weak spot: 68% found outside recording 151. Earlier: `ball_v2` on data-team's held-out test recordings: **0.793 recall** / 0.890 precision at conf 0.20, 30 FPS on CPU (42 FPS as ONNX). 80% of misses are one stationary ball repeated in 113 frames of SPL recording 151; counting it once, recall is 0.947. `ball_v3_960` and `ball_v2_aug` didn't beat it. Next: `ball_v4`, with recording 180 added to test for a real tiny-ball check. See [detection results](../detection-team/README.md#results) |
| Evaluation & benchmark harness (testing-deployment-team) | Hossein, Henry | `eval_harness.py` runs in CI on every PR and gates on `benchmarks/baseline.json`; baseline reflects a real trained model, not just the dummy detector | Oct 15 | **In progress** — harness done (precision, recall, mAP@0.5, ms/frame; runs in CI on every PR). A pretrained-COCO-YOLO stand-in detector is ready for an interim run on real images. **Waiting on** a real labeled val set (data-team) and a trained model (detection-team) before the CI gate means anything |
| Multi-class detection (detection-team) | Suhaas, Aditya Mitra | ≥80% recall @ IoU 0.5 across ball, robot, goalpost and landmark (field_line was taken out of the box detector — team decision, Oct 15) | Nov 30 | **In progress** — `multiclass_v1` (placeholder split): 0.786 recall overall, AP 0.86 ball / 0.82 goalpost / 0.76 robot / 0.73 landmark. `multiclass_v2` on data-team's split is the real check |
| Ground-plane projection (integration-team) | Phil, Henry | Robot-relative position error ≤ 0.3 m at 2 m range, measured against a known checkerboard/reference-point layout | Nov 30 | Not started |
| ROS 2 perception integration (integration-team) | Phil, Henry | Perception messages published at ≥10 Hz with the schema documented in [ARCHITECTURE.md](../ARCHITECTURE.md#data-contracts) | Dec 15 | Not started |
| Final documentation & demo | Whole team | A new member can reproduce the pipeline end to end from README + ONBOARDING alone | Dec 15 | Not started |

## Template row

Copy this when adding a milestone:

```
| <module> | <owner(s)> | <measurable target, with a number> | <date> | <Not started / In progress / Done> |
```
