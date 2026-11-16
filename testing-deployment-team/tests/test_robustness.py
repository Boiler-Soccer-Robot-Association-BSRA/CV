"""Tests for testing-deployment-team/tests/robustness.py. No model runs:
the detector is a fake whose answers depend on the image it is shown."""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import robustness  # noqa: E402

from shared.types import BoundingBox, Detection  # noqa: E402

BALL_BOX = [10.0, 10.0, 30.0, 30.0]
ROBOT_BOX = [40.0, 5.0, 60.0, 45.0]


def sample_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(60, 200, (50, 80, 3), dtype=np.uint8)


def apply(name: str, image: np.ndarray, boxes=(), seed: int = 0) -> np.ndarray:
    return robustness.CONDITIONS_BY_NAME[name].transform(image, list(boxes), random.Random(seed))


def test_every_condition_keeps_the_frame_contract_and_is_repeatable():
    image = sample_image()
    original = image.copy()
    for condition in robustness.CONDITIONS:
        if condition.transform is None:
            assert condition.name == "clean"
            continue
        first = apply(condition.name, image, [BALL_BOX])
        assert first.shape == image.shape and first.dtype == np.uint8, condition.name
        assert not np.array_equal(first, image), f"{condition.name} changed nothing"
        assert np.array_equal(first, apply(condition.name, image, [BALL_BOX])), condition.name
    assert np.array_equal(image, original), "a condition modified its input in place"


def test_lighting_conditions_move_brightness_the_right_way():
    image = sample_image()
    mean = image.mean()
    assert apply("dark", image).mean() < apply("dim", image).mean() < mean
    assert apply("bright", image).mean() > mean
    assert apply("low_contrast", image).std() < image.std() * 0.6
    warm, cool = apply("warm", image), apply("cool", image)
    assert warm[:, :, 2].mean() > image[:, :, 2].mean() > cool[:, :, 2].mean()  # red (BGR)
    assert warm[:, :, 0].mean() < image[:, :, 0].mean() < cool[:, :, 0].mean()  # blue


def test_motion_blur_smears_horizontally_and_more_with_length():
    image = np.zeros((21, 41, 3), dtype=np.uint8)
    image[:, 20] = 255  # one bright vertical line

    def spread(name: str) -> int:
        return int((apply(name, image)[10, :, 0] > 0).sum())

    assert spread("motion_5") < spread("motion_9") < spread("motion_15")
    assert spread("motion_15") >= 13
    # A horizontal line blurred horizontally stays one row thick.
    image = np.zeros((21, 41, 3), dtype=np.uint8)
    image[10, :] = 255
    assert int((apply("motion_15", image)[:, 20, 0] > 0).sum()) == 1


@pytest.mark.parametrize(("name", "fraction"), [("occluded_25", 0.25), ("occluded_50", 0.50)])
def test_occlusion_covers_the_stated_share_of_each_box_and_nothing_else(name, fraction):
    image = np.full((50, 80, 3), 255, dtype=np.uint8)
    for seed in range(8):  # different seeds pick different sides
        out = apply(name, image, [BALL_BOX, ROBOT_BOX], seed)
        covered = (out == robustness.OCCLUDER_GRAY).all(axis=2)
        for x0, y0, x1, y1 in ([int(v) for v in BALL_BOX], [int(v) for v in ROBOT_BOX]):
            share = covered[y0:y1, x0:x1].mean()
            assert share == pytest.approx(fraction, abs=0.03)
        outside = covered.copy()
        outside[10:30, 10:30] = False
        outside[5:45, 40:60] = False
        assert not outside.any()


def test_occluder_rect_sides():
    assert robustness.occluder_rect([0, 0, 10, 20], 0.5, "left") == (0, 0, 5, 20)
    assert robustness.occluder_rect([0, 0, 10, 20], 0.5, "right") == (5, 0, 10, 20)
    assert robustness.occluder_rect([0, 0, 10, 20], 0.25, "top") == (0, 0, 10, 5)
    assert robustness.occluder_rect([0, 0, 10, 20], 0.25, "bottom") == (0, 15, 10, 20)
    with pytest.raises(ValueError):
        robustness.occluder_rect([0, 0, 10, 20], 0.5, "middle")


class BrightnessDetector:
    """Finds the robot always, the ball only in frames that aren't dark,
    and raises one false ball alarm per frame."""

    def predict(self, image: np.ndarray) -> list[Detection]:
        detections = [
            Detection("robot", 0.9, BoundingBox(*ROBOT_BOX)),
            Detection("ball", 0.3, BoundingBox(70, 40, 78, 48)),
        ]
        if image.mean() > 60:
            detections.append(Detection("ball", 0.8, BoundingBox(*BALL_BOX)))
        return detections


def write_dataset(folder: Path, names: list[str]) -> list[dict]:
    manifest = []
    for name in names:
        cv2.imwrite(str(folder / name), sample_image())
        manifest.append(
            {
                "image": name,
                "annotations": [
                    {"class_name": "ball", "bbox": BALL_BOX},
                    {"class_name": "robot", "bbox": ROBOT_BOX},
                ],
            }
        )
    return manifest


def test_evaluate_condition_counts_recall_per_class(tmp_path):
    manifest = write_dataset(tmp_path, ["a.png", "b.png", "c.png", "d.png"])
    clean = robustness.evaluate_condition(
        manifest, tmp_path, BrightnessDetector(), robustness.CONDITIONS_BY_NAME["clean"]
    )
    assert clean["per_class"]["ball"] == {
        "labeled": 4,
        "found": 4,
        "false_alarms": 4,
        "recall": 1.0,
        "precision": 0.5,
    }
    assert clean["per_class"]["robot"]["recall"] == 1.0
    assert clean["recall"] == 1.0 and clean["labeled"] == 8

    dark = robustness.evaluate_condition(
        manifest, tmp_path, BrightnessDetector(), robustness.CONDITIONS_BY_NAME["dark"]
    )
    assert dark["per_class"]["ball"]["recall"] == 0.0
    assert dark["per_class"]["robot"]["recall"] == 1.0
    assert dark["recall"] == 0.5
    assert "count_once" not in dark


def test_count_once_collapses_one_recording_into_one_object(tmp_path):
    # Three frames of recording 151 (the same ball each time) and two others.
    names = ["151-a.png", "151-b.png", "151-c.png", "97-a.png", "97-b.png"]
    manifest = write_dataset(tmp_path, names)

    class MissesRecording151(BrightnessDetector):
        def __init__(self, find_one: bool) -> None:
            self.frame, self.find_one = 0, find_one

        def predict(self, image):
            self.frame += 1
            detections = super().predict(image)
            if self.frame <= 3 and not (self.find_one and self.frame == 2):
                detections = [d for d in detections if d.confidence != 0.8]
            return detections

    clean = robustness.CONDITIONS_BY_NAME["clean"]
    never = robustness.evaluate_condition(
        manifest, tmp_path, MissesRecording151(False), clean, count_once=("151-", "ball")
    )
    assert never["per_class"]["ball"]["recall"] == pytest.approx(2 / 5)  # as scored
    assert never["count_once"]["labeled"] == 3 and never["count_once"]["found"] == 2
    assert never["count_once"]["recall"] == pytest.approx(2 / 3)

    once = robustness.evaluate_condition(
        manifest, tmp_path, MissesRecording151(True), clean, count_once=("151-", "ball")
    )
    assert once["per_class"]["ball"]["recall"] == pytest.approx(3 / 5)
    assert once["count_once"]["found"] == 3 and once["count_once"]["recall"] == 1.0


def test_examples_are_saved_per_condition(tmp_path):
    manifest = write_dataset(tmp_path, [f"{i}.png" for i in range(6)])
    examples = tmp_path / "examples"
    robustness.evaluate_condition(
        manifest,
        tmp_path,
        BrightnessDetector(),
        robustness.CONDITIONS_BY_NAME["occluded_50"],
        examples_dir=examples,
        num_examples=3,
    )
    assert sorted(p.name for p in examples.iterdir()) == [
        "occluded_50__0.jpg",
        "occluded_50__2.jpg",
        "occluded_50__4.jpg",
    ]


def test_table_shows_recall_and_change_from_clean(tmp_path):
    manifest = write_dataset(tmp_path, ["151-a.png", "b.png"])
    results = [
        robustness.evaluate_condition(
            manifest,
            tmp_path,
            BrightnessDetector(),
            robustness.CONDITIONS_BY_NAME[name],
            count_once=("151-", "ball"),
        )
        for name in ("clean", "dark")
    ]
    lines = robustness.format_table(results).splitlines()
    assert lines[0] == "| Condition | What changes | ball | robot | ball (counted once) | all |"
    assert lines[2].startswith("| `clean` | unchanged test images | 1.000 | 1.000 | 1.000 | 1.000")
    assert "0.000 (-100.0) | 1.000 (+0.0) | 0.000 (-100.0) | 0.500 (-50.0)" in lines[3]


def test_parse_count_once():
    assert robustness.parse_count_once("151-:ball") == ("151-", "ball")
    with pytest.raises(Exception, match="PREFIX:CLASS"):
        robustness.parse_count_once("151")


def test_augment_random_lighting_still_matches_its_fixed_form():
    """augment.py's adjust_lighting was split so the fixed form could be
    reused here; the random one must draw and apply the same values."""
    image = sample_image()
    rng = random.Random(7)
    expected = robustness.augment.apply_lighting(
        image,
        contrast=rng.uniform(0.6, 1.4),
        brightness=rng.uniform(-40, 40),
        warmth=rng.uniform(-0.12, 0.12),
        gamma=rng.uniform(0.6, 1.6),
    )
    assert np.array_equal(robustness.augment.adjust_lighting(image, random.Random(7)), expected)
    assert np.array_equal(robustness.augment.apply_lighting(image), image)
