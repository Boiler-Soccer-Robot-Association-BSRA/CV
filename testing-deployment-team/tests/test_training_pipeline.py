"""Tests for detection-team's training pipeline: the TORSO-21 placeholder
split, train.py's config handling and provenance, and eval.py's YOLO →
harness-manifest conversion. Uses a tiny fake TORSO-21 tree and a fake
YOLO model — no dataset download, no training."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "detection-team" / "training"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TRAINING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


placeholder = load("placeholder_dataset")
train_module = load("train")
eval_module = load("eval")


# --- placeholder_dataset.py -------------------------------------------------


def torso_image(width=100, height=80, balls=(), other=()) -> dict:
    annotations = [
        {"type": "ball", "in_image": True, "vector": [list(p1), list(p2)]} for p1, p2 in balls
    ]
    annotations += [{"type": t, "in_image": True, "vector": [[0, 0], [5, 5]]} for t in other]
    annotations.append({"type": "ball", "in_image": False})
    return {"width": width, "height": height, "annotations": annotations}


def test_ball_boxes_keeps_visible_balls_clips_and_drops_slivers():
    image = torso_image(
        balls=[((10, 20), (30, 40)), ((-5, 70), (20, 95)), ((50, 50), (51, 60))],
        other=["robot"],
    )
    assert placeholder.ball_boxes(image) == [(10.0, 20.0, 30.0, 40.0), (0.0, 70.0, 20.0, 80.0)]


def test_yolo_label_lines_are_normalized_center_format_with_shared_class_ids():
    lines = placeholder.yolo_label_lines(
        [("ball", (10.0, 20.0, 30.0, 60.0)), ("landmark", (0.0, 0.0, 10.0, 8.0))], 100, 80
    )
    assert lines == [
        "0 0.200000 0.500000 0.200000 0.500000",
        "4 0.050000 0.050000 0.100000 0.100000",
    ]


def test_object_boxes_maps_torso_labels_to_our_classes():
    image = {
        "width": 200,
        "height": 100,
        "annotations": [
            {"type": "goalpost", "in_image": True, "vector": [[10, 90], [20, 88], [15, 5], [8, 6]]},
            {"type": "T-Intersection", "in_image": True, "vector": [[100, 50]]},
            {"type": "robot", "in_image": True, "vector": [[30, 30], [60, 80]]},
            {"type": "obstacle", "in_image": True, "vector": [[0, 0], [9, 9]]},
        ],
    }
    objects = placeholder.object_boxes(image, ("robot", "goalpost", "landmark"))
    half = placeholder.LANDMARK_BOX_FRACTION * 200 / 2
    assert objects == [
        ("goalpost", (8.0, 5.0, 20.0, 90.0)),
        ("landmark", (100 - half, 50 - half, 100 + half, 50 + half)),
        ("robot", (30.0, 30.0, 60.0, 80.0)),
    ]
    assert placeholder.object_boxes(image, ("ball",)) == []


def test_assign_recordings_keeps_each_recording_in_one_split_and_is_seeded():
    counts = {str(i): 10 for i in range(20)}
    a = placeholder.assign_recordings(counts, val_fraction=0.1, test_fraction=0.2, seed=3)
    assert a == placeholder.assign_recordings(counts, 0.1, 0.2, seed=3)
    assert set(a) == set(counts)
    by_split = {s: sum(counts[r] for r in a if a[r] == s) for s in ("train", "val", "test")}
    assert by_split == {"train": 140, "val": 20, "test": 40}


def test_multiclass_build_uses_the_same_recordings_as_ball_only(tmp_path):
    torso = tmp_path / "TORSO-21"
    make_fake_torso(torso)
    ball = placeholder.build(torso, tmp_path / "b", 0.25, 0.25, seed=0, negatives_per_positive=1)
    multi = placeholder.build(
        torso, tmp_path / "m", 0.25, 0.25, 0, 1, classes=("ball", "robot", "goalpost", "landmark")
    )
    for split in ("train", "val", "test"):
        assert ball["splits"][split]["recordings"] == multi["splits"][split]["recordings"]
    names = yaml.safe_load((tmp_path / "m" / "data.yaml").read_text())["names"]
    assert names == {0: "ball", 1: "robot", 2: "goalpost", 3: "field_line", 4: "landmark"}


def make_fake_torso(root: Path) -> None:
    """Two halves like the real download; recording "1" appears in both, as
    real TORSO-21 recordings do."""
    layout = {
        "train": {f"{rec}-img{i:03d}.png": rec for rec in ("1", "2", "3", "4") for i in range(3)},
        "test": {f"1-img9{i:02d}.png": "1" for i in range(2)},
    }
    for half, files in layout.items():
        base = root / f"reality_{half}" / half
        (base / "images").mkdir(parents=True)
        entries = {}
        for i, name in enumerate(sorted(files)):
            cv2.imwrite(str(base / "images" / name), np.zeros((80, 100, 3), np.uint8))
            balls = [((10, 10), (30, 30))] if i % 3 else []
            entries[name] = torso_image(balls=balls)
        (base / "annotations.yaml").write_text(yaml.safe_dump({"images": entries}))


def test_build_writes_yolo_dataset_split_by_recording(tmp_path):
    torso = tmp_path / "TORSO-21"
    make_fake_torso(torso)
    out = tmp_path / "placeholder"
    info = placeholder.build(torso, out, 0.25, 0.25, seed=0, negatives_per_positive=10)

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "ball"}
    seen = {}
    for split in ("train", "val", "test"):
        images = sorted((out / "images" / split).iterdir())
        assert all(p.is_symlink() and p.resolve().exists() for p in images)
        assert len(images) == len(list((out / "labels" / split).iterdir()))
        for p in images:
            assert seen.setdefault(placeholder.recording_id(p.name), split) == split
    # Recording 1's frames from both TORSO halves land together.
    assert len([n for n in seen if n == "1"]) == 1
    assert sum(c["images"] for c in info["splits"].values()) == 14


def test_build_refuses_to_overwrite(tmp_path):
    torso = tmp_path / "TORSO-21"
    make_fake_torso(torso)
    out = tmp_path / "placeholder"
    placeholder.build(torso, out, 0.25, 0.25, seed=0, negatives_per_positive=1)
    with pytest.raises(SystemExit, match="already exists"):
        placeholder.build(torso, out, 0.25, 0.25, seed=0, negatives_per_positive=1)


# --- eval.py ------------------------------------------------------------------


def test_label_path_for_swaps_the_last_images_dir(tmp_path):
    image = tmp_path / "images_backup" / "images" / "test" / "a.png"
    expected = tmp_path / "images_backup" / "labels" / "test" / "a.txt"
    assert eval_module.label_path_for(image) == expected


def test_yolo_split_to_manifest_round_trips_placeholder_boxes(tmp_path):
    torso = tmp_path / "TORSO-21"
    make_fake_torso(torso)
    out = tmp_path / "placeholder"
    # No val/test, so every recording (and every ball) lands in train.
    placeholder.build(torso, out, 0.0, 0.0, seed=0, negatives_per_positive=10)

    manifest = eval_module.yolo_split_to_manifest(out / "data.yaml", "train")
    assert len(manifest) == 14
    boxes = [a for entry in manifest for a in entry["annotations"]]
    assert boxes and all(a["class_name"] == "ball" for a in boxes)
    assert all(a["bbox"] == pytest.approx([10, 10, 30, 30], abs=1e-3) for a in boxes)
    assert all(Path(entry["image"]).is_absolute() for entry in manifest)


def test_yolo_split_to_manifest_can_keep_only_some_classes(tmp_path):
    root = tmp_path / "ds"
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "test" / "a.png"), np.zeros((50, 100, 3), np.uint8))
    (root / "labels" / "test" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n")
    (root / "data.yaml").write_text(
        yaml.safe_dump({"path": str(root), "test": "images/test", "names": ["ball", "robot"]})
    )
    everything = eval_module.yolo_split_to_manifest(root / "data.yaml", "test")
    balls = eval_module.yolo_split_to_manifest(root / "data.yaml", "test", ["ball"])
    assert [a["class_name"] for a in everything[0]["annotations"]] == ["ball", "robot"]
    assert balls[0]["annotations"] == [
        {"class_name": "ball", "bbox": pytest.approx([40.0, 20.0, 60.0, 30.0])}
    ]


# --- train.py -----------------------------------------------------------------


def write_config(tmp_path: Path, **overrides) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "data.yaml").write_text("names: {0: ball}\n")
    config = {
        "name": "unit_test_model",
        "model": "yolov8n.pt",
        "dataset": "data/data.yaml",
        "epochs": 1,
        "imgsz": 64,
        "batch": 2,
        "seed": 0,
        **overrides,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_resolve_config_makes_paths_relative_to_the_config(tmp_path):
    path = write_config(tmp_path, model="weights/start.pt")
    config = train_module.resolve_config(path)
    assert config["dataset"] == str((tmp_path / "data" / "data.yaml").resolve())
    assert config["model"] == str((tmp_path / "weights" / "start.pt").resolve())


def test_resolve_config_leaves_stock_model_names_alone(tmp_path):
    config = train_module.resolve_config(write_config(tmp_path, model="yolov8n.pt"))
    assert config["model"] == "yolov8n.pt"


def test_resolve_config_reports_missing_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\n")
    with pytest.raises(SystemExit, match="missing required keys"):
        train_module.resolve_config(path)


class FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        if kwargs.get("resume"):  # resuming: the run folder is where last.pt lives
            save_dir = Path(self.weights).parent.parent
        else:
            save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"fake weights")
        self.trainer = SimpleNamespace(save_dir=save_dir)
        return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5})


def test_train_copies_best_weights_and_writes_provenance(tmp_path, monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train_module, "REPO_ROOT", tmp_path)
    config_path = write_config(tmp_path, device="cpu", train_args={"plots": False})

    weights = train_module.train(config_path)

    assert weights == tmp_path / "models" / "unit_test_model.pt"
    assert weights.read_bytes() == b"fake weights"
    provenance = json.loads(weights.with_suffix(".json").read_text())
    assert provenance["config"]["epochs"] == 1
    assert provenance["final_val_metrics"] == {"metrics/mAP50(B)": 0.5}
    assert provenance["dataset"]["data"] == {"names": {"0": "ball"}}  # JSON keys are strings
    assert {"commit", "dirty"} <= set(provenance["git"])

    with pytest.raises(SystemExit, match="already exists"):
        train_module.train(config_path)


def test_pick_device_respects_explicit_choice():
    assert train_module.pick_device("cpu") == "cpu"
    assert train_module.pick_device(None) in ("mps", None)


def test_resume_continues_from_last_checkpoint(tmp_path, monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train_module, "REPO_ROOT", tmp_path)
    config_path = write_config(tmp_path)

    with pytest.raises(SystemExit, match="Nothing to resume"):
        train_module.train(config_path, resume=True)

    last = tmp_path / "runs" / "unit_test_model" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"half trained")
    weights = train_module.train(config_path, resume=True)
    assert json.loads(weights.with_suffix(".json").read_text())["resumed"] is True
