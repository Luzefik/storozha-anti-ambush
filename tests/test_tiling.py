"""SAHI tiling on synthetic frames: no gated data, runs in CI."""

import numpy as np
import pytest

from zfpv.data.tiling import boxes_in_tile, nms, predict_tiled, tile_origins


def test_tiles_cover_frame_and_last_tile_ends_at_border():
    origins = tile_origins(1280, 720, tile=640, overlap=0.2)
    assert sorted({x for x, _ in origins}) == [0, 512, 640]
    assert sorted({y for _, y in origins}) == [0, 80]
    assert len(origins) == 6


def test_frame_smaller_than_tile_gets_one_tile_on_that_axis():
    assert tile_origins(576, 1024, tile=640, overlap=0.2) == [(0, 0), (0, 384)]


@pytest.mark.parametrize("tile, overlap", [(0, 0.2), (640, 1.0)])
def test_bad_arguments_raise(tile, overlap):
    with pytest.raises(ValueError):
        tile_origins(1280, 720, tile, overlap)


def test_boxes_are_shifted_clipped_and_filtered():
    boxes = [
        [600, 100, 620, 120],  # inside tile x0=512
        [500, 100, 540, 140],  # 70% inside -> kept, clipped
        [505, 100, 515, 110],  # 30% inside -> dropped
        [100, 100, 120, 120],  # outside
    ]
    kept = boxes_in_tile(boxes, x0=512, y0=0, tile=640, min_visible=0.5)
    np.testing.assert_allclose(kept, [[88, 100, 108, 120], [0, 100, 28, 140]])
    assert boxes_in_tile(np.zeros((0, 4)), 0, 0, 640, 0.5).shape == (0, 4)


def test_nms_keeps_best_of_overlapping_and_both_of_separate():
    kept = nms([[0, 0, 10, 10, 0.6], [1, 1, 11, 11, 0.9], [50, 50, 60, 60, 0.5]], 0.5)
    np.testing.assert_allclose(kept[:, 4], [0.9, 0.5])


def test_predict_tiled_maps_tiles_back_and_merges_duplicates():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    shapes = []

    def predict(images):  # every image reports a "drone" at its own (10, 10)
        shapes.extend(im.shape[:2] for im in images)
        return [[[10, 10, 20, 20, 0.5]] for _ in images]

    dets = predict_tiled(predict, frame, tile=640)
    assert shapes.count((640, 640)) == 6 and shapes.count((720, 1280)) == 1
    # tile (0, 0) and the full frame see the same spot -> merged; 5 other tiles stay
    assert len(dets) == 6
    assert dets[:, 2].max() <= 1280 and dets[:, 3].max() <= 720
