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

`main` is protected: every pull request needs **at least one approving
review from someone other than the author**, plus a passing CI check,
before it can merge. [.github/CODEOWNERS](.github/CODEOWNERS) lists who
owns each folder — GitHub will suggest them as reviewers automatically.
The author-can't-approve-their-own-PR part is enforced by GitHub itself
once branch protection is turned on; see the repo's branch protection
settings (Settings → Branches) if a PR seems to be missing that
requirement.

Budget time for review rather than expecting an instant merge — this is
a six-person team, not a solo project.
