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
| Dev environment & tooling | Whole team | All 6 members run `./scripts/setup.sh` and `python scripts/quickstart.py` successfully | Sept 30 | Not started |
| Ball detection (detection-team) | Suhaas, Aditya Mitra | 90% recall @ IoU 0.5 on a held-out RoboCup validation split, ≥30 FPS on a MacBook (CPU) | Oct 31 | Not started |
| Evaluation & benchmark harness (testing-deployment-team) | Hossein | `eval_harness.py` runs in CI on every PR and gates on `benchmarks/baseline.json`; baseline reflects a real trained model, not just the dummy detector | Oct 15 | **In progress** — harness implemented, still running against the dummy detector; needs a real model + real dataset to set a meaningful baseline |
| Multi-class detection (detection-team) | Suhaas, Aditya Mitra | ≥80% recall @ IoU 0.5 across all five classes (ball, robot, goalpost, field_line, landmark) | Nov 30 | Not started |
| Ground-plane projection (integration-team) | Phil | Robot-relative position error ≤ 0.3 m at 2 m range, measured against a known checkerboard/reference-point layout | Nov 30 | Not started |
| ROS 2 perception integration (integration-team) | Phil | Perception messages published at ≥10 Hz with the schema documented in [ARCHITECTURE.md](../ARCHITECTURE.md#data-contracts) | Dec 15 | Not started |
| Final documentation & demo | Whole team | A new member can reproduce the pipeline end to end from README + ONBOARDING alone | Dec 15 | Not started |

## Template row

Copy this when adding a milestone:

```
| <module> | <owner(s)> | <measurable target, with a number> | <date> | <Not started / In progress / Done> |
```
