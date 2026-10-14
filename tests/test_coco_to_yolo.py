"""coco_to_yolo on tiny synthetic COCO: no gated data, runs in CI."""

import json

import pytest

from zfpv.data.coco_to_yolo import coco_json_to_yolo

W, H = 1280, 720

# video-11 frame_000005 (image_id 6), real box, by hand:
#   x_c = (392.23 + 29.21 / 2) / 1280 = 406.835 / 1280 = 0.317840
#   y_c = (614.17 + 31.23 / 2) / 720  = 629.785 / 720  = 0.874701
#   w_n = 29.21 / 1280 = 0.022820
#   h_n = 31.23 / 720  = 0.043375
VIDEO_11_FRAME_5_BOX = [392.23, 614.17, 29.21, 31.23]
VIDEO_11_FRAME_5_LINE = "0 0.317840 0.874701 0.022820 0.043375"


def _write_coco(tmp_path, images, annotations):
    coco = {
        "images": [
            {"id": i, "file_name": f"frame_{i - 1:06d}.png", "width": W, "height": H}
            for i in images
        ],
        "annotations": [
            {"id": k, "image_id": image_id, "category_id": 1, "bbox": bbox}
            for k, (image_id, bbox) in enumerate(annotations, start=1)
        ],
        "categories": [{"id": 1, "name": "drone"}],
    }
    path = tmp_path / "instances_default.json"
    path.write_text(json.dumps(coco))
    return path


def _values(line):
    cls, *xywh = line.split()
    return int(cls), [float(v) for v in xywh]


def test_known_box_converts_to_known_yolo_line(tmp_path):
    coco = _write_coco(tmp_path, images=[6], annotations=[(6, VIDEO_11_FRAME_5_BOX)])
    out = tmp_path / "labels"

    coco_json_to_yolo(coco, out)

    assert (out / "frame_000005.txt").read_text().strip() == VIDEO_11_FRAME_5_LINE


def test_every_image_gets_a_file_and_negatives_are_empty(tmp_path):
    coco = _write_coco(tmp_path, images=[1, 2], annotations=[(1, VIDEO_11_FRAME_5_BOX)])
    out = tmp_path / "labels"

    n = coco_json_to_yolo(coco, out)

    assert n == 2
    assert sorted(p.name for p in out.iterdir()) == ["frame_000000.txt", "frame_000001.txt"]
    assert (out / "frame_000001.txt").read_text() == ""


def test_box_past_right_and_bottom_edge_is_clipped(tmp_path):
    # sticks out right + bottom: visible x 1260..1280, y 700..720
    coco = _write_coco(tmp_path, images=[1], annotations=[(1, [1260.0, 700.0, 40.0, 30.0])])
    out = tmp_path / "labels"

    coco_json_to_yolo(coco, out)

    cls, (x_c, y_c, w_n, h_n) = _values((out / "frame_000000.txt").read_text())
    assert cls == 0
    assert x_c == pytest.approx(1270 / W, abs=1e-6)
    assert y_c == pytest.approx(710 / H, abs=1e-6)
    assert w_n == pytest.approx(20 / W, abs=1e-6)
    assert h_n == pytest.approx(20 / H, abs=1e-6)
    # 6 decimals -> edge can be 1.0000005
    assert x_c + w_n / 2 <= 1.0 + 1e-6 and y_c + h_n / 2 <= 1.0 + 1e-6


def test_box_with_negative_corner_is_clipped(tmp_path):
    # like video-2: x_min < 0
    coco = _write_coco(tmp_path, images=[1], annotations=[(1, [-2.0, 10.0, 22.0, 20.0])])
    out = tmp_path / "labels"

    coco_json_to_yolo(coco, out)

    _, (x_c, _, w_n, _) = _values((out / "frame_000000.txt").read_text())
    assert x_c - w_n / 2 == pytest.approx(0.0, abs=1e-6)
    assert w_n == pytest.approx(20 / W, abs=1e-6)


def test_box_entirely_outside_the_frame_is_dropped(tmp_path):
    coco = _write_coco(
        tmp_path,
        images=[1],
        annotations=[(1, [1300.0, 10.0, 20.0, 20.0]), (1, VIDEO_11_FRAME_5_BOX)],
    )
    out = tmp_path / "labels"

    coco_json_to_yolo(coco, out)

    lines = (out / "frame_000000.txt").read_text().strip().splitlines()
    assert lines == [VIDEO_11_FRAME_5_LINE]
