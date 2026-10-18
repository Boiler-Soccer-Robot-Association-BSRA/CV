"""Tests for data-team/scripts/split.py and augment.py.

Run from the repo root:  pytest data-team/tests
They build a tiny fake TORSO-21 folder, so no download is needed.
"""

import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import augment  # noqa: E402
from yolo_io import read_labels, write_labels  # noqa: E402


def make_fake_torso(torso: Path) -> None:
    """TORSO-21/reality/{train,test} where test/ reuses train/'s recordings."""
    rng = random.Random(0)
    for split, frames in (("train", range(0, 8)), ("test", range(8, 10))):
        d = torso / "reality" / split
        (d / "images").mkdir(parents=True)
        images = {}
        for rec in range(100, 160):  # 60 recordings, each in BOTH train/ and test/
            for i in frames:
                name = f"{rec}-img{i:05d}.png"
                cv2.imwrite(str(d / "images" / name), np.zeros((48, 64, 3), np.uint8))
                anns = [
                    {"type": "robot", "in_image": True, "vector": [[1, 1], [20, 30]]},
                    {"type": "T-Intersection", "in_image": True, "vector": [[32, 24]]},
                ]
                kind = i % 5
                if kind < 3:  # ball present
                    x, y = rng.randint(0, 50), rng.randint(0, 30)
                    ball = {"type": "ball", "in_image": True, "vector": [[x, y], [x + 8, y + 8]]}
                    anns.append(ball)
                elif kind == 3:  # ball confirmed absent
                    anns.append({"type": "ball", "in_image": False})
                # kind 4: ball not mentioned -> unknown, never used as a negative
                images[name] = {"width": 64, "height": 48, "annotations": anns}
        (d / "annotations.yaml").write_text(yaml.safe_dump({"images": images}))


def run_split(torso: Path, out: Path, *extra) -> dict:
    cmd = [sys.executable, str(SCRIPTS / "split.py"), "--torso-dir", str(torso), "--out", str(out)]
    subprocess.run([*cmd, *extra], check=True, capture_output=True)
    return yaml.safe_load((out / "split_info.yaml").read_text())


@pytest.fixture
def torso(tmp_path):
    make_fake_torso(tmp_path / "TORSO-21")
    return tmp_path / "TORSO-21"


def image_lists(out: Path) -> dict:
    return {s: sorted(p.name for p in (out / "images" / s).iterdir()) for s in SPLITS}


SPLITS = ("train", "val", "test")


def test_split_is_reproducible(torso, tmp_path):
    a = run_split(torso, tmp_path / "a")
    b = run_split(torso, tmp_path / "b")
    assert a["fingerprint"] == b["fingerprint"]
    assert image_lists(tmp_path / "a") == image_lists(tmp_path / "b")
    c = run_split(torso, tmp_path / "c", "--seed", "7")
    assert c["fingerprint"] != a["fingerprint"]


def test_no_recording_in_two_splits(torso, tmp_path):
    info = run_split(torso, tmp_path / "out")
    recs = {s: set(info["splits"][s]["recordings"]) for s in SPLITS}
    assert all(recs.values())
    assert not recs["train"] & recs["val"]
    assert not recs["train"] & recs["test"]
    assert not recs["val"] & recs["test"]
    lists = image_lists(tmp_path / "out")
    for split in SPLITS:
        assert {n.split("-")[0] for n in lists[split]} == recs[split]
    # TORSO-21's test/ frames (00008, 00009) are pooled with train/
    everything = [n for names in lists.values() for n in names]
    assert any(n.endswith(("00008.png", "00009.png")) for n in everything)
    # 60 recordings x 6 ball images, plus ~10% confirmed-empty (rounded per split);
    # "unknown" images (ball not labeled either way) are never used
    out = tmp_path / "out"
    with_ball = [p for p in (out / "labels").glob("*/*.txt") if p.read_text().strip()]
    assert len(with_ball) == 360
    assert 34 <= len(everything) - 360 <= 38


def test_ball_and_multiclass_share_test_recordings(torso, tmp_path):
    ball = run_split(torso, tmp_path / "ball")
    multi = run_split(torso, tmp_path / "multi", "--classes", "ball", "robot", "landmark")
    for split in SPLITS:
        assert ball["splits"][split]["recordings"] == multi["splits"][split]["recordings"]


def test_data_yaml_follows_contract(torso, tmp_path):
    run_split(torso, tmp_path / "ball")
    data = yaml.safe_load((tmp_path / "ball" / "data.yaml").read_text())
    assert data["names"] == {0: "ball"}
    assert (data["train"], data["val"], data["test"]) == (
        "images/train",
        "images/val",
        "images/test",
    )
    run_split(torso, tmp_path / "multi", "--classes", "landmark", "ball")
    data = yaml.safe_load((tmp_path / "multi" / "data.yaml").read_text())
    assert data["names"] == {0: "ball", 1: "robot", 2: "goalpost", 3: "field_line", 4: "landmark"}


def test_labels_are_ball_only_yolo(torso, tmp_path):
    run_split(torso, tmp_path / "out")
    for lbl in (tmp_path / "out" / "labels" / "train").glob("*.txt"):
        for line in lbl.read_text().splitlines():
            cid, *vals = line.split()
            assert cid == "0"  # ball; robots are not included in the ball subset
            assert all(0 <= float(v) <= 1 for v in vals)


def test_landmarks_get_square_boxes(torso, tmp_path):
    run_split(torso, tmp_path / "out", "--classes", "landmark")
    lines = []
    for lbl in (tmp_path / "out" / "labels").glob("*/*.txt"):
        lines += lbl.read_text().splitlines()
    assert lines
    for line in lines:
        cid, cx, cy, w, h = line.split()
        assert cid == "4"  # landmark
        assert float(cx) == pytest.approx(0.5) and float(cy) == pytest.approx(0.5)
        assert float(w) == pytest.approx(0.05, abs=1e-5)  # 5% of image width
        assert float(h) * 48 == pytest.approx(float(w) * 64, abs=1e-3)  # square in pixels


def test_field_line_is_not_offered(torso, tmp_path):
    cmd = [sys.executable, str(SCRIPTS / "split.py"), "--torso-dir", str(torso)]
    result = subprocess.run([*cmd, "--classes", "field_line"], capture_output=True, check=False)
    assert result.returncode != 0


def test_crop_scale_moves_boxes_with_the_image():
    img = np.zeros((120, 160, 3), np.uint8)
    img[40:60, 70:90] = 255  # a white "ball"
    boxes = np.array([[70, 40, 90, 60]], dtype=float)
    rng = random.Random(1)
    for _ in range(50):
        out, keep, new = augment.crop_scale(img, boxes, rng)
        assert out.shape == img.shape
        if len(keep):
            x0, y0, x1, y1 = new[0].round().astype(int)
            inside = out[y0:y1, x0:x1]
            assert inside.mean() > 150  # the box still sits on the white square


def test_augment_writes_matching_labels(torso, tmp_path):
    out = tmp_path / "out"
    run_split(torso, out)
    cmd = [sys.executable, str(SCRIPTS / "augment.py"), str(out), "--copies", "2"]
    subprocess.run(cmd, check=True, capture_output=True)
    n_train = len(list((out / "images" / "train").iterdir()))
    assert len(list((out / "images" / "train_aug").iterdir())) == 2 * n_train
    assert len(list((out / "labels" / "train_aug").iterdir())) == 2 * n_train
    data = yaml.safe_load((out / "data_aug.yaml").read_text())
    assert data["train"] == ["images/train", "images/train_aug"]
    bad = subprocess.run(
        [sys.executable, str(SCRIPTS / "augment.py"), str(out), "--split", "val"],
        capture_output=True,
        check=False,
    )
    assert bad.returncode != 0


def test_yolo_roundtrip(tmp_path):
    boxes = np.array([[10.0, 20.0, 30.0, 50.0]])
    write_labels(tmp_path / "x.txt", [0], boxes, 100, 80)
    ids, back = read_labels(tmp_path / "x.txt", 100, 80)
    assert ids == [0]
    assert np.allclose(back, boxes, atol=1e-3)
