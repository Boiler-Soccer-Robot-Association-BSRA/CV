# Contributing

Quick reference for how work actually gets merged. See
[ONBOARDING.md](ONBOARDING.md) for first-time setup and
[README.md](README.md#team-norms) for team norms in general.

## Before you start

```bash
./scripts/setup.sh
source venv/bin/activate
```

This installs the pinned dependencies (`requirements.txt`,
`requirements-dev.txt`) and the pre-commit hooks — see
[Pre-commit hooks](#pre-commit-hooks) below.

## Branching

Name feature branches `<subteam>/<short-task-name>`:

- `data-team/torso21-download`
- `detection-team/baseline-yolo-train`
- `integration-team/camera-calibration`
- `testing-deployment-team/eval-harness`

Never commit directly to `main`.

## Before opening a pull request

```bash
ruff check .                              # lint
ruff format .                             # auto-format
pytest testing-deployment-team/tests/     # unit tests
python testing-deployment-team/tests/eval_harness.py   # benchmark regression gate
```

All four run in CI (`.github/workflows/lint-test.yml`) on every pull
request, but running them locally first saves a round trip.

## Pre-commit hooks

`./scripts/setup.sh` runs `pre-commit install` for you, so `ruff` lint
and format fixes, and a few basic file hygiene checks, run automatically
on `git commit`. If a hook fails, it typically means it already fixed
the file in place — `git add` the result and commit again. To run the
hooks manually against everything (e.g. after pulling changes that
predate you installing them): `pre-commit run --all-files`.

## Opening the pull request

- Fill out the [PR template](.github/pull_request_template.md) — what
  changed, which subteam/area, whether you tested it locally.
- Keep datasets and model weights out of the diff (`.gitignore` already
  excludes the usual spots) — share those through the team drive folder.
- If you touch a stage of the pipeline described in
  [ARCHITECTURE.md](ARCHITECTURE.md#pipeline-status), update that
  stage's row in the same PR.

## Review

`main` is protected: a pull request can merge once its CI check
(`lint-test`, from `.github/workflows/lint-test.yml`) passes. **An
approving review is not required** — if CI is green, you can merge your
own PR. Direct pushes to `main` are still off; everything goes through
a PR so CI runs and there's a record of what changed.

Reviews are still encouraged. [.github/CODEOWNERS](.github/CODEOWNERS)
lists who owns each folder, and GitHub requests them automatically. If
your PR touches another subteam's folder or `shared/`, tag its owners
on the PR or post in the group chat before merging — you don't have to
wait for them, but they shouldn't find out from a broken import. If
someone leaves review comments after you've merged, address them in a
follow-up PR.

These rules are enforced by the repo's branch protection settings
(Settings → Branches, admin access needed); if GitHub is blocking a
merge that this section says should be allowed, the settings and this
doc have drifted — tell the Computer Vision Lead.
