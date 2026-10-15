"""Train a YOLO model on a labeled dataset split.

    python detection-team/training/train.py detection-team/configs/ball_v1.yaml

Everything about a run lives in its config (detection-team/configs/,
loaded via shared.config.load_config) — nothing is hardcoded here. When
training finishes, the best checkpoint is copied to
detection-team/models/<name>.pt (gitignored; share it via the team
drive) with a <name>.json next to it recording exactly what produced it:
the config, the dataset's split info, the git commit, library versions,
and the final validation metrics. Ultralytics' own run folder (plots,
per-epoch CSV, last.pt) is kept under detection-team/runs/train/<name>/.

Config keys:

    name: ball_v1                 # output is models/<name>.pt; must be unique
    model: yolov8n.pt             # starting weights (stock name, or a path relative to the config)
    dataset: ../datasets/placeholder_ball/data.yaml   # relative to the config file
    epochs: 40
    imgsz: 640
    batch: 16
    seed: 0
    patience: 10                  # stop early after this many epochs without val improvement
    device: null                  # null = auto (Apple GPU if present), or "cpu", "mps", "0", ...
    train_args: {}                # any other ultralytics train() arguments, passed through
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.config import load_config  # noqa: E402

MODELS_DIR = REPO_ROOT / "detection-team" / "models"
RUNS_DIR = REPO_ROOT / "detection-team" / "runs" / "train"
REQUIRED_KEYS = ("name", "model", "dataset", "epochs", "imgsz", "batch", "seed")


def resolve_config(config_path: Path) -> dict[str, Any]:
    """Load a training config and make its paths absolute (relative paths
    are relative to the config file)."""
    config = load_config(config_path)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise SystemExit(f"{config_path} is missing required keys: {', '.join(missing)}")

    base = config_path.resolve().parent
    config["dataset"] = str((base / config["dataset"]).resolve())
    # "yolov8n.pt" is a stock name ultralytics downloads; anything with a
    # directory in it is a local checkpoint.
    if Path(config["model"]).parent != Path("."):
        config["model"] = str((base / config["model"]).resolve())
    config.setdefault("patience", 100)
    config.setdefault("device", None)
    config.setdefault("train_args", {})
    return config


def pick_device(device: str | None) -> str | None:
    """Ultralytics never picks the Apple GPU on its own (it falls back to
    the CPU, ~2.5x slower on an M-series Mac), so choose it when `device`
    is left unset. Otherwise defer to ultralytics (CUDA if present, else CPU)."""
    if device is not None:
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else None


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def dataset_info(data_yaml: Path) -> dict[str, Any]:
    """What we know about the dataset: its data.yaml, plus split_info.yaml
    if the dataset was built with one (placeholder_dataset.py writes it)."""
    info: dict[str, Any] = {"data_yaml": str(data_yaml)}
    info["data"] = yaml.safe_load(data_yaml.read_text())
    split_info = data_yaml.parent / "split_info.yaml"
    if split_info.exists():
        info["split_info"] = yaml.safe_load(split_info.read_text())
    return info


def versions() -> dict[str, str]:
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def train(config_path: Path, overwrite: bool = False) -> Path:
    config = resolve_config(config_path)
    name = config["name"]
    weights_out = MODELS_DIR / f"{name}.pt"
    if weights_out.exists() and not overwrite:
        raise SystemExit(
            f"{weights_out} already exists. Give the config a new `name`, or pass --overwrite."
        )
    data_yaml = Path(config["dataset"])
    if not data_yaml.exists():
        raise SystemExit(f"Dataset not found: {data_yaml}")

    from ultralytics import YOLO

    model = YOLO(config["model"])
    started = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        seed=config["seed"],
        deterministic=True,
        patience=config["patience"],
        device=pick_device(config["device"]),
        project=str(RUNS_DIR),
        name=name,
        exist_ok=overwrite,
        **config["train_args"],
    )
    duration_s = time.time() - started

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, weights_out)

    provenance = {
        "name": name,
        "weights": str(weights_out.relative_to(REPO_ROOT)),
        "config_path": str(config_path),
        "config": config,
        "dataset": dataset_info(data_yaml),
        "git": git_state(),
        "versions": versions(),
        "train_duration_s": round(duration_s, 1),
        "run_dir": str(save_dir),
        "final_val_metrics": {
            k: float(v) for k, v in (getattr(results, "results_dict", None) or {}).items()
        },
    }
    weights_out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    return weights_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path, help="Path to a training config YAML.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace models/<name>.pt if it exists."
    )
    args = parser.parse_args()

    weights = train(args.config, args.overwrite)
    print(f"Saved {weights} (and {weights.with_suffix('.json').name} with how it was made)")


if __name__ == "__main__":
    main()
