# Milestones

Per-module owner, measurable target, and target date, so "done" means
something concrete instead of "it runs."

**This is a draft.** The rows below are seeded from the semester
timeline already in [README.md](../README.md#semester-timeline) and
each subteam's roadmap — the targets are illustrative starting points,
not commitments anyone has actually signed off on yet. Each subteam
should confirm or adjust its own rows (owner, exact metric, date) and
open a PR updating this file once they have.

Last reviewed: **Nov 16, 2026**.

Update this table whenever a milestone is hit, missed, or rescheduled —
a stale milestones doc is worse than none, same rule as the
[Pipeline status](../ARCHITECTURE.md#pipeline-status) table.

| Module | Owner(s) | Target metric | Target date | Status |
|---|---|---|---|---|
| Dev environment & tooling | Whole team | All 7 members run `./scripts/setup.sh` and `python scripts/quickstart.py` successfully | Sept 30 | **Done** — `setup.sh` and `quickstart.py` exist and CI installs the environment; closed by the CV lead Nov 8 |
| Ball detection (detection-team) | Suhaas, Aditya Mitra | 90% recall @ IoU 0.5 on a held-out RoboCup validation split, ≥30 FPS on a MacBook (CPU) | Nov 15 (moved from Oct 31) | **Done (borderline)** — team decision Nov 8: counts as met, and both recall numbers are always reported. `ball_v4` on the test set with recording 180 held out (867 balls after dropping 4 labels with no visible ball, see `data-team/label_fixes.yaml`): **0.915 recall** counting recording 151's stationary ball once (**0.803 as scored**), 0.921 precision at conf 0.20, 31 FPS on CPU (43 FPS as ONNX). Tiny balls are the weak spot: 42 of 58 (72%) found outside recording 151. `ball_v2`, `ball_v3_960` and `ball_v2_aug` didn't beat it. See [detection results](../detection-team/README.md#results) |
| Evaluation & benchmark harness (testing-deployment-team) | Hossein, Henry | `eval_harness.py` runs in CI on every PR and gates on `benchmarks/baseline.json`; baseline reflects a real trained model, not just the dummy detector | Oct 15 | **Done Nov 16** (a month late) — the harness runs in CI on every PR and gates on `benchmarks/baseline.json`, which now holds `multiclass_v2`'s scores on the held-out test set (1,405 frames, conf 0.20): 0.853 recall / 0.834 precision, mAP 0.827; per class as scored: ball 0.790, goalpost 0.864, landmark 0.857, robot 0.876. CI scores the model's recorded detections (weights and TORSO-21 stay out of git) and fails if recall drops more than 0.05 overall or for any class. Limit: CI runs no inference, so the recording must be redone when the model, test split or `Detector` changes. See [Real-model regression gate](../testing-deployment-team/README.md#real-model-regression-gate) |
| Multi-class detection (detection-team) | Suhaas, Aditya Mitra | ≥80% recall @ IoU 0.5 across ball, robot, goalpost and landmark (field_line was taken out of the box detector — team decision, Oct 15) | Nov 30 | **Done** (decision, Nov 12: the target holds per class, with the ball target's count-once rule) — `multiclass_v2` on data-team's split with recording 180 held out (1,405 frames, 6,806 objects), conf 0.20: **0.853 recall** / 0.834 precision, mAP 0.827, 32 FPS on CPU (44 FPS as ONNX). Per class: robot 0.876, goalpost 0.864, landmark 0.857, ball **0.907 counting recording 151's stationary ball once (0.790 as scored)**. `multiclass_v2` is the model for the robot. See [detection results](../detection-team/README.md#multiclass_v2-recording-180-held-out-for-testing) |
| Ground-plane projection (integration-team) | Phil, Henry | Robot-relative position error ≤ 0.3 m at 2 m range, measured against a known checkerboard/reference-point layout | Nov 30 | **In progress** — `projection.py` is implemented and unit-tested on a synthetic camera (including lens distortion). Still to do: calibrate the real camera, measure its mount, and check positions against a taped floor grid for the 0.3 m @ 2 m target |
| ROS 2 perception integration (integration-team) | Phil, Henry | Perception messages published at ≥10 Hz with the schema documented in [ARCHITECTURE.md](../ARCHITECTURE.md#data-contracts) | Dec 15 | **In progress** — the message schema is defined (four `.msg` files, #33, matching `shared/types.py`) and `camera_node.py` / `perception_node.py` are written (Nov 16). Tested without ROS only: the detect → project → message path runs real footage at 26–28 frames/s on a MacBook CPU (M3 Max, `multiclass_v2.pt`; 42–46 as dynamic ONNX), above the 10 Hz target. Still to do: run the nodes inside ROS 2 and measure the published rate with `ros2 topic hz`, then on the robot's computer |
| Final documentation & demo | Whole team | A new member can reproduce the pipeline end to end from README + ONBOARDING alone | Dec 15 | **In progress** — setup, data, training and evaluation are documented; the end-to-end part depends on projection and ROS 2 |

## Template row

Copy this when adding a milestone:

```
| <module> | <owner(s)> | <measurable target, with a number> | <date> | <Not started / In progress / Done> |
```
