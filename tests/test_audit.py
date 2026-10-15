"""Audit helpers on tiny synthetic inputs: no gated data, runs in CI."""

import json
import zipfile

import pandas as pd
import pytest

from zfpv.data.audit import box_table, size_shift_test


def _label_zip(raw_dir, video, width, height, bbox):
    coco = {
        "images": [{"id": 1, "file_name": "frame_000000.png", "width": width, "height": height}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": bbox}],
        "categories": [{"id": 1, "name": "drone"}],
    }
    (raw_dir / "labels").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_dir / "labels" / f"spie-{video}.zip", "w") as zf:
        zf.writestr("annotations/instances_default.json", json.dumps(coco))


def test_letterbox_scales_by_the_long_side(tmp_path):
    # 40x40 box: 1280x720 -> x0.5, 576x1024 -> x0.625
    _label_zip(tmp_path, "video-1", 1280, 720, [0, 0, 40, 40])
    _label_zip(tmp_path, "video-2", 576, 1024, [0, 0, 40, 40])

    boxes = box_table(tmp_path, imgsz=640).set_index("video")

    assert boxes.loc["video-1", "size_lb"] == pytest.approx(20.0)
    assert boxes.loc["video-2", "size_lb"] == pytest.approx(25.0)
    assert boxes.loc["video-1", "size_bucket"] == "16-32"


def test_size_shift_test_counts_every_video_assignment():
    # 4 train + 2 val -> C(6, 2) = 15 picks. real: 31.5 - 1.5 = 30
    # no other pick that extreme (val = {32, 33}: 15.5 - 32.5 = -17) -> p = 1/15
    per_video = pd.DataFrame(
        {
            "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
            "size_lb_med": [30.0, 31.0, 32.0, 33.0, 1.0, 2.0, 30.0, 31.0],
        }
    )

    result = size_shift_test(per_video).set_index("compare")

    val = result.loc["train vs val"]
    assert val["n_permutations"] == 15
    assert val["p_two_sided"] == pytest.approx(1 / 15)
