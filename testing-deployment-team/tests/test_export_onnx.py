"""Tests for testing-deployment-team/deployment/export_onnx.py. Nothing is
exported here (CI has no trained weights): the export and the evaluation
are replaced with fakes, and the logic around them is checked."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "testing-deployment-team" / "deployment" / "export_onnx.py"
spec = importlib.util.spec_from_file_location("export_onnx", SCRIPT)
export_onnx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_onnx)


def metrics(precision: float, recall: float, map50: float = 0.8) -> dict:
    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "ms_per_frame": 25.0,
        "num_frames": 10,
        "num_predictions": 42,
    }


def test_trained_imgsz_reads_the_checkpoint_first():
    model = SimpleNamespace(ckpt={"train_args": {"imgsz": 960}}, overrides={"imgsz": 640})
    assert export_onnx.trained_imgsz(model) == 960


def test_trained_imgsz_falls_back_to_overrides_and_handles_lists():
    assert export_onnx.trained_imgsz(SimpleNamespace(ckpt=None, overrides={"imgsz": 640})) == 640
    model = SimpleNamespace(ckpt={"train_args": {"imgsz": [480, 640]}}, overrides={})
    assert export_onnx.trained_imgsz(model) == 640


def test_trained_imgsz_refuses_to_guess():
    with pytest.raises(ValueError, match="--imgsz"):
        export_onnx.trained_imgsz(SimpleNamespace(ckpt={}, overrides={}))


def test_export_rejects_wrong_file_types(tmp_path):
    with pytest.raises(ValueError, match=".pt"):
        export_onnx.export(tmp_path / "model.onnx", tmp_path / "out.onnx")
    with pytest.raises(ValueError, match=".onnx"):
        export_onnx.export(tmp_path / "model.pt", tmp_path / "out.pt")


def test_compare_passes_within_tolerance_and_fails_outside():
    close = export_onnx.compare(metrics(0.834, 0.853), metrics(0.822, 0.850), max_diff=0.02)
    assert close["passed"]
    assert close["diff"]["precision"] == pytest.approx(-0.012)
    assert close["diff"]["recall"] == pytest.approx(-0.003)
    far = export_onnx.compare(metrics(0.834, 0.853), metrics(0.834, 0.800), max_diff=0.02)
    assert not far["passed"]


def test_compare_does_not_gate_on_map():
    result = export_onnx.compare(metrics(0.9, 0.9, 0.9), metrics(0.9, 0.9, 0.5), max_diff=0.02)
    assert result["passed"]
    assert result["diff"]["map50"] == pytest.approx(-0.4)


def run_main(tmp_path, monkeypatch, onnx_recall: float, extra: list[str] | None = None) -> int:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [ball]\n")
    calls = {}

    def fake_export(weights, output, imgsz, dynamic):
        calls["export"] = (imgsz, dynamic)
        return {
            "weights": str(weights),
            "output": str(output),
            "imgsz": 640,
            "dynamic": dynamic,
            "input_shape": [1, 3, 640, 640],
            "opset": 18,
            "size_mb": 12.3,
        }

    def fake_check(weights, output, data_yaml, split, conf, iou, max_diff, work_dir):
        calls["check"] = (split, conf)
        pt, onnx_ = metrics(0.9, 0.9), metrics(0.9, onnx_recall)
        return {
            "dataset": str(data_yaml),
            "split": split,
            "conf": conf,
            "iou_threshold": iou,
            "device": "cpu",
            "pt": pt,
            "onnx": onnx_,
            **export_onnx.compare(pt, onnx_, max_diff),
        }

    monkeypatch.setattr(export_onnx, "export", fake_export)
    monkeypatch.setattr(export_onnx, "check_export", fake_check)
    argv = ["export_onnx.py", str(weights), "--output", str(tmp_path / "model.onnx")]
    argv += ["--results", str(tmp_path / "results.json"), "--check", str(data_yaml)]
    monkeypatch.setattr(sys, "argv", argv + (extra or []))
    code = export_onnx.main()
    run_main.calls = calls
    return code


def test_main_passes_when_the_export_matches(tmp_path, monkeypatch, capsys):
    assert run_main(tmp_path, monkeypatch, onnx_recall=0.895) == 0
    assert run_main.calls == {"export": (None, False), "check": ("test", 0.20)}
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["export"]["opset"] == 18
    assert results["check"]["passed"]
    assert "PASS" in capsys.readouterr().out


def test_main_exits_nonzero_when_the_export_disagrees(tmp_path, monkeypatch, capsys):
    assert run_main(tmp_path, monkeypatch, onnx_recall=0.7, extra=["--dynamic"]) == 1
    assert run_main.calls["export"] == (None, True)
    assert "FAIL" in capsys.readouterr().out
