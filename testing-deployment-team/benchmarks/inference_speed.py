"""Benchmark inference speed of a trained model on CPU.

Dec Week 11. Times `Detector.predict` (detection-team/inference/detector.py)
over N runs and reports avg / median / p95 milliseconds and FPS. Works for
`.pt` and `.onnx` weights; give several to compare them in one session:

    python testing-deployment-team/benchmarks/inference_speed.py \\
        detection-team/models/multiclass_v2.pt detection-team/models/multiclass_v2.onnx \\
        --images data-team/datasets/multiclass_r180/images/test --runs 300

**What is timed.** One `predict()` call per run: resize + inference + NMS +
conversion to `Detection`s, which is what the robot pays per frame. Reading
and decoding the image is not timed (frames are loaded into memory first),
and neither are the `--warmup` calls made before timing starts (the first
calls are slow while the runtime allocates buffers).

**Which frames.** `--images` is a folder (or one image); the runs cycle
through its frames in name order, taking evenly spaced ones if the folder
has more than `--runs`. Speed depends on frame size and on how many objects
are in view, so use real frames from the test set for numbers you report.
With no `--images` it uses the tiny synthetic fixture frames, which is only
good for checking the script runs.

**FPS** is 1000 / avg ms. It is the detector alone on this machine; the
robot's onboard computer will differ. Always report the device with the
number. `.onnx` needs `onnxruntime` (testing-deployment-team/requirements.txt).

Results are printed and written to --output as JSON (gitignored).
"""

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detection-team" / "inference"))

DEFAULT_IMAGES = REPO_ROOT / "testing-deployment-team" / "tests" / "fixtures" / "sample_dataset"
DEFAULT_OUTPUT = (
    REPO_ROOT / "testing-deployment-team" / "benchmarks" / "results" / "inference_speed.json"
)
DEFAULT_RUNS = 100
DEFAULT_WARMUP = 10
DEFAULT_CONF = 0.20  # the operating point in detection-team's results
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_frames(path: Path, limit: int) -> list[np.ndarray]:
    """Up to `limit` BGR frames from an image file or a folder of images,
    evenly spaced through the folder (name order) so one recording doesn't
    dominate."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No images found in {path}")
    if len(files) > limit:
        files = [files[i * len(files) // limit] for i in range(limit)]
    frames = []
    for file in files:
        frame = cv2.imread(str(file))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {file}")
        frames.append(frame)
    return frames


def time_predict(
    predict: Callable[[np.ndarray], Any],
    frames: Sequence[np.ndarray],
    runs: int = DEFAULT_RUNS,
    warmup: int = DEFAULT_WARMUP,
) -> list[float]:
    """Call `predict` `warmup` times untimed, then `runs` times timed, cycling
    through `frames`. Returns each timed call's wall-clock milliseconds."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if not frames:
        raise ValueError("need at least one frame")
    for i in range(warmup):
        predict(frames[i % len(frames)])
    times_ms = []
    for i in range(runs):
        frame = frames[i % len(frames)]
        start = time.perf_counter()
        predict(frame)
        times_ms.append((time.perf_counter() - start) * 1000)
    return times_ms


def summarize(times_ms: Sequence[float]) -> dict[str, float | int]:
    """avg / median / p95 / min / max milliseconds and FPS (1000 / avg)."""
    if not times_ms:
        raise ValueError("no timings to summarize")
    avg = statistics.fmean(times_ms)
    return {
        "runs": len(times_ms),
        "avg_ms": avg,
        "median_ms": statistics.median(times_ms),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "fps": 1000 / avg,
    }


def environment(device: str) -> dict[str, Any]:
    """What the numbers were measured on, to store next to them."""
    import torch

    info = {
        "device": device,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import onnxruntime

        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        info["onnxruntime"] = None
    return info


def benchmark_weights(
    weights: Path,
    frames: Sequence[np.ndarray],
    runs: int,
    warmup: int,
    conf: float,
    device: str,
) -> dict[str, Any]:
    from detector import Detector

    detector = Detector(weights, conf=conf, device=device)
    summary = summarize(time_predict(detector.predict, frames, runs, warmup))
    return {"weights": str(weights), "format": weights.suffix.lstrip("."), **summary}


def format_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        f"{'weights':<28} {'avg ms':>8} {'median':>8} {'p95':>8} {'FPS':>7}",
    ]
    for row in rows:
        lines.append(
            f"{Path(row['weights']).name:<28} {row['avg_ms']:>8.1f} {row['median_ms']:>8.1f} "
            f"{row['p95_ms']:>8.1f} {row['fps']:>7.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path, nargs="+", help="One or more .pt / .onnx files.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Timed predict() calls.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Untimed calls first.")
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Folder of frames (or one image) to run on. Default: the synthetic fixture frames.",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default="cpu", help='Default "cpu" (the milestone\'s device).')
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing = [str(w) for w in args.weights if not w.exists()]
    if missing:
        parser.error(f"weights not found: {', '.join(missing)}")

    frames = load_frames(args.images, args.runs)
    sizes = sorted({f"{f.shape[1]}x{f.shape[0]}" for f in frames})
    print(
        f"{len(frames)} frames from {args.images} ({', '.join(sizes)}); "
        f"{args.runs} runs after {args.warmup} warmup; conf {args.conf}; device {args.device}"
    )

    rows = [
        benchmark_weights(w, frames, args.runs, args.warmup, args.conf, args.device)
        for w in args.weights
    ]
    print(format_table(rows))

    results = {
        "results": rows,
        "runs": args.runs,
        "warmup": args.warmup,
        "conf": args.conf,
        "images": str(args.images),
        "num_frames": len(frames),
        "frame_sizes": sizes,
        "environment": environment(args.device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
