# Onboarding

Welcome to BSRA-CV! This is the order to read things in and get set up.

## 1. Read, in this order

1. [README.md](README.md) — what we're building, the four subteams,
   setup instructions, semester timeline.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — how the subteams' work fits
   into one pipeline, the exact data types passed between stages, and
   the **Pipeline status** table showing what's actually built versus
   still just planned. Read that table closely — a lot of the repo
   right now is scaffolding, not working code, and the table says
   exactly which parts.
3. Your subteam's own README — its slice of the semester roadmap,
   folder layout, and "First task" section to actually start on:
   - [data-team/README.md](data-team/README.md)
   - [detection-team/README.md](detection-team/README.md)
   - [integration-team/README.md](integration-team/README.md)
   - [testing-deployment-team/README.md](testing-deployment-team/README.md)
4. [GLOSSARY.md](GLOSSARY.md) — if a term in any of the above is
   unfamiliar (TORSO-21, mAP, ROS 2 node, ONNX, ...), it's probably
   defined here.

## 2. One-time setup

```bash
git clone https://github.com/hhatamiy/BSRA-CV.git
cd BSRA-CV
./scripts/setup.sh          # creates venv/, installs deps, installs pre-commit hooks
source venv/bin/activate    # do this in every new shell
python scripts/quickstart.py
```

If `quickstart.py` prints a handful of detections, your environment
works. Read its docstring — it's honest about what it does and doesn't
prove about the pipeline. If your subteam needs anything beyond the
root `requirements.txt` (ROS 2, ONNX/TensorRT), that's called out in
the "Setup" section of your subteam's README.

## 3. Say hello

Check [.github/CODEOWNERS](.github/CODEOWNERS) for who's on your
subteam. If it's still showing placeholder usernames instead of real
ones, message the Computer Vision Lead or Programming Director listed
in the root README's [Leadership section](README.md#leadership) instead
— they'll point you to the right people.

Then go do your subteam's "First task." Questions and blockers go in
the group chat between meetings rather than sitting stuck — see Team
norms in the root README.
