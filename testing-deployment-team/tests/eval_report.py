"""Visual report for the eval harness: what the detector *should* have
found (ground truth) next to what it *did* find, frame by frame.

    python testing-deployment-team/tests/eval_harness.py --report-dir /tmp/report
    open /tmp/report/index.html

Each frame is drawn with:

  - correct detection (true positive): solid green box, "ball 0.87"
  - false alarm (false positive):      solid red box,   "FP ball 0.42"
  - missed object (false negative):    dashed orange box, "MISSED ball"
  - ground truth that was found:       thin white box under the green one

The page lists the run's metrics, then every frame — failures first —
with buttons to show only frames with misses, only false alarms, or only
fully correct frames. Matching is exactly the harness's own
(`match_frame_indices`), so what you see is what the numbers counted.
"""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.types import Detection  # noqa: E402

# BGR
CORRECT_COLOR = (60, 200, 60)
FALSE_ALARM_COLOR = (60, 60, 230)
MISSED_COLOR = (0, 165, 255)
GROUND_TRUTH_COLOR = (255, 255, 255)
OUTLINE_COLOR = (0, 0, 0)

# Report images are resized so a big run stays a reasonable size and tiny
# frames (like the 64px fixtures) are still big enough to read labels on.
REPORT_MAX_SIDE = 960
REPORT_MIN_SIDE = 480


def _scaled_box(bbox: list[float] | Any, scale: float) -> tuple[int, int, int, int]:
    if isinstance(bbox, (list, tuple)):
        x_min, y_min, x_max, y_max = bbox
    else:
        x_min, y_min, x_max, y_max = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
    return tuple(int(round(v * scale)) for v in (x_min, y_min, x_max, y_max))


def _dashed_rectangle(canvas, top_left, bottom_right, color, thickness, dash: int) -> None:
    (x1, y1), (x2, y2) = top_left, bottom_right
    for start, end, fixed, horizontal in (
        (x1, x2, y1, True),
        (x1, x2, y2, True),
        (y1, y2, x1, False),
        (y1, y2, x2, False),
    ):
        for a in range(start, end, dash * 2):
            b = min(a + dash, end)
            p, q = ((a, fixed), (b, fixed)) if horizontal else ((fixed, a), (fixed, b))
            cv2.line(canvas, p, q, color, thickness)


def _label(canvas, text: str, x: int, y: int, color, font_scale: float, thickness: int) -> None:
    height, width = canvas.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    top = y - text_h - baseline
    if top < 0:
        top = max(y, 0)
    left = min(max(x, 0), max(width - text_w, 0))
    cv2.rectangle(canvas, (left, top), (left + text_w, top + text_h + baseline), color, cv2.FILLED)
    cv2.putText(
        canvas,
        text,
        (left, top + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        OUTLINE_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_evaluation(
    image: np.ndarray,
    ground_truth: list[dict],
    matches: list[tuple[Detection, int | None]],
    missed: list[int],
    max_side: int | None = None,
    min_side: int | None = None,
) -> np.ndarray:
    """Return a copy of `image` with ground truth and predictions drawn
    and colored by outcome (see module docstring). `matches` and
    `missed` are what eval_harness.match_frame_indices returns. If
    `max_side`/`min_side` are set, the image's longer side is resized
    into that range before drawing, so line widths and labels stay
    readable."""
    height, width = image.shape[:2]
    longest = max(height, width)
    scale = 1.0
    if max_side and longest > max_side:
        scale = max_side / longest
    elif min_side and longest < min_side:
        scale = min_side / longest
    if scale != 1.0:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, size, interpolation=interpolation)
    canvas = image.copy()
    short_side = min(canvas.shape[:2])
    thickness = max(2, round(short_side / 300))
    font_scale = max(0.4, short_side / 900)
    text_thickness = max(1, thickness // 2)

    found = {idx for _, idx in matches if idx is not None}
    for idx in sorted(found):
        x1, y1, x2, y2 = _scaled_box(ground_truth[idx]["bbox"], scale)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), OUTLINE_COLOR, thickness + 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GROUND_TRUTH_COLOR, max(1, thickness // 2))

    for idx in missed:
        gt = ground_truth[idx]
        x1, y1, x2, y2 = _scaled_box(gt["bbox"], scale)
        _dashed_rectangle(canvas, (x1, y1), (x2, y2), MISSED_COLOR, thickness, dash=thickness * 3)
        _label(
            canvas, f"MISSED {gt['class_name']}", x1, y1, MISSED_COLOR, font_scale, text_thickness
        )

    # Lowest confidence first, so the most confident boxes end up on top.
    for pred, idx in sorted(matches, key=lambda m: m[0].confidence):
        x1, y1, x2, y2 = _scaled_box(pred.bbox, scale)
        if idx is None:
            color, text = FALSE_ALARM_COLOR, f"FP {pred.class_name} {pred.confidence:.2f}"
        else:
            color, text = CORRECT_COLOR, f"{pred.class_name} {pred.confidence:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        _label(canvas, text, x1, y1, color, font_scale, text_thickness)
    return canvas


def frame_tags(record: dict[str, Any]) -> list[str]:
    """'missed' and/or 'false_alarm', or ['correct'] if neither."""
    tags = []
    if record["missed"]:
        tags.append("missed")
    if any(idx is None for _, idx in record["matches"]):
        tags.append("false_alarm")
    return tags or ["correct"]


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_report(
    results: dict[str, Any],
    frame_records: list[dict[str, Any]],
    out_dir: Path,
    detector_name: str = "",
    weights: str | None = None,
    manifest: str = "",
) -> Path:
    """Write annotated frames to out_dir/frames/ and an index.html that
    shows them alongside the run's metrics. Returns the index.html path."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for i, record in enumerate(frame_records):
        image = cv2.imread(str(record["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {record['image_path']}")
        drawn = draw_evaluation(
            image,
            record["ground_truth"],
            record["matches"],
            record["missed"],
            REPORT_MAX_SIDE,
            REPORT_MIN_SIDE,
        )
        filename = f"{i:05d}_{Path(record['image_path']).stem}.jpg"
        cv2.imwrite(str(frames_dir / filename), drawn, [cv2.IMWRITE_JPEG_QUALITY, 85])

        tp = sum(1 for _, idx in record["matches"] if idx is not None)
        fp = len(record["matches"]) - tp
        cards.append(
            {
                "file": f"frames/{filename}",
                "name": Path(record["image_path"]).name,
                "tags": frame_tags(record),
                "tp": tp,
                "fp": fp,
                "missed": len(record["missed"]),
                "gt": len(record["ground_truth"]),
                "ms": record["ms"],
            }
        )

    # Failures first (most mistakes first), then correct frames, each in manifest order.
    cards.sort(key=lambda c: (c["tags"] == ["correct"], -(c["fp"] + c["missed"])))

    index = out_dir / "index.html"
    index.write_text(
        _render_html(results, cards, detector_name, weights, manifest), encoding="utf-8"
    )
    return index


def _render_html(
    results: dict[str, Any],
    cards: list[dict[str, Any]],
    detector_name: str,
    weights: str | None,
    manifest: str,
) -> str:
    esc = html.escape
    counts = {
        tag: sum(1 for c in cards if tag in c["tags"])
        for tag in ("missed", "false_alarm", "correct")
    }
    ms = results.get("ms_per_frame") or 0.0
    fps = f"{1000 / ms:.1f}" if ms > 0 else "n/a"
    per_class = "".join(
        f"<li>{esc(name)}: {ap:.3f}</li>"
        for name, ap in (results.get("per_class_ap") or {}).items()
    )
    iou = results.get("iou_threshold", 0.5)

    tag_labels = {"missed": "missed", "false_alarm": "false alarm", "correct": "correct"}
    card_html = []
    for c in cards:
        chips = "".join(f'<span class="chip {tag}">{tag_labels[tag]}</span>' for tag in c["tags"])
        card_html.append(
            f'<figure class="card" data-tags="{" ".join(c["tags"])}">'
            f'<a href="{esc(c["file"])}" target="_blank">'
            f'<img loading="lazy" src="{esc(c["file"])}" alt="{esc(c["name"])}"></a>'
            f'<figcaption><div class="name">{esc(c["name"])}</div>'
            f'<div class="stats">{c["tp"]} correct · {c["fp"]} false alarm · '
            f"{c['missed']} missed · {c['gt']} expected · {c['ms']:.0f} ms</div>"
            f"<div>{chips}</div></figcaption></figure>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval report</title>
<style>
  :root {{
    --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #5d6677; --border: #dde1e8;
    --correct: #2f9e44; --false-alarm: #d9363e; --missed: #e8890c; --accent: #3b5bdb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14171c; --panel: #1d2128; --text: #e6e9ef; --muted: #9aa3b2; --border: #2e343e;
      --accent: #748ffc;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
         font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  main {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .meta {{ color: var(--muted); margin-bottom: 20px; word-break: break-all; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
             gap: 10px; margin-bottom: 16px; }}
  .metric {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
            padding: 10px 12px; }}
  .metric .label {{ color: var(--muted); font-size: 12px; }}
  .metric .value {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: 12px 0 20px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; }}
  .swatch {{ width: 22px; height: 14px; border: 3px solid; border-radius: 2px; }}
  .filters button {{ background: var(--panel); color: var(--text); border: 1px solid var(--border);
                    border-radius: 999px; padding: 6px 12px; cursor: pointer; font: inherit; }}
  .filters button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 600; }}
  .grid {{ display: grid; gap: 14px;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }}
  .card {{ margin: 0; background: var(--panel); border: 1px solid var(--border);
          border-radius: 8px; overflow: hidden; }}
  .card img {{ display: block; width: 100%; height: auto; background: #000; }}
  figcaption {{ padding: 8px 10px 10px; }}
  .name {{ font-weight: 600; word-break: break-all; }}
  .stats {{ color: var(--muted); font-size: 12px; margin: 2px 0 6px; }}
  .chip {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 8px;
          border-radius: 999px; margin-right: 4px; color: #fff; }}
  .chip.correct {{ background: var(--correct); }}
  .chip.false_alarm {{ background: var(--false-alarm); }}
  .chip.missed {{ background: var(--missed); }}
  details {{ margin-bottom: 8px; color: var(--muted); }}
  .empty {{ color: var(--muted); padding: 24px 0; }}
</style>
</head>
<body>
<main>
  <h1>Eval report</h1>
  <div class="meta">
    {esc(detector_name)}{" · weights " + esc(weights) if weights else ""} · {esc(manifest)}<br>
    generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
  </div>

  <div class="metrics">
    <div class="metric"><div class="label">Precision</div>
      <div class="value">{_fmt(results.get("precision"))}</div></div>
    <div class="metric"><div class="label">Recall</div>
      <div class="value">{_fmt(results.get("recall"))}</div></div>
    <div class="metric"><div class="label">mAP @ IoU {iou}</div>
      <div class="value">{_fmt(results.get("map50"))}</div></div>
    <div class="metric"><div class="label">ms / frame (FPS)</div>
      <div class="value">{ms:.1f} ({fps})</div></div>
    <div class="metric"><div class="label">Frames</div>
      <div class="value">{results.get("num_frames", len(cards))}</div></div>
    <div class="metric"><div class="label">Expected objects</div>
      <div class="value">{results.get("num_ground_truth", "n/a")}</div></div>
    <div class="metric"><div class="label">Predictions</div>
      <div class="value">{results.get("num_predictions", "n/a")}</div></div>
  </div>
  {f"<details><summary>Per-class AP</summary><ul>{per_class}</ul></details>" if per_class else ""}

  <div class="row legend">
    <span><i class="swatch" style="border-color: var(--correct)"></i>correct detection</span>
    <span><i class="swatch" style="border-color: var(--false-alarm)"></i>false alarm (FP)</span>
    <span><i class="swatch" style="border-color: var(--missed); border-style: dashed"></i>
      missed (MISSED)</span>
    <span><i class="swatch" style="border-color: #fff; outline: 1px solid #000"></i>
      ground truth that was found</span>
  </div>

  <div class="row filters">
    <button data-filter="all" class="active">All ({len(cards)})</button>
    <button data-filter="failures">Failures ({len(cards) - counts["correct"]})</button>
    <button data-filter="missed">Missed ({counts["missed"]})</button>
    <button data-filter="false_alarm">False alarms ({counts["false_alarm"]})</button>
    <button data-filter="correct">Correct ({counts["correct"]})</button>
  </div>

  <div class="grid">
    {"".join(card_html)}
  </div>
  <p class="empty" hidden>No frames match this filter.</p>
</main>
<script>
  const buttons = document.querySelectorAll(".filters button");
  const cards = document.querySelectorAll(".card");
  const empty = document.querySelector(".empty");
  buttons.forEach((button) => button.addEventListener("click", () => {{
    buttons.forEach((b) => b.classList.toggle("active", b === button));
    const filter = button.dataset.filter;
    let shown = 0;
    cards.forEach((card) => {{
      const tags = card.dataset.tags.split(" ");
      const show = filter === "all"
        || (filter === "failures" ? !tags.includes("correct") : tags.includes(filter));
      card.hidden = !show;
      if (show) shown++;
    }});
    empty.hidden = shown > 0;
  }}));
</script>
</body>
</html>
"""
