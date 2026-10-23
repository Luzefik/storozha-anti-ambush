from __future__ import annotations

import numpy as np


def _starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    return sorted({*range(0, length - tile, step), length - tile})  # last tile ends at the border


def tile_origins(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """Top-left (x0, y0) of every tile, covering the whole frame including the edges."""
    if tile <= 0 or not 0.0 <= overlap < 1.0:
        raise ValueError(f"need tile > 0 and 0 <= overlap < 1, got {tile}, {overlap}")
    step = max(1, round(tile * (1 - overlap)))
    return [(x, y) for y in _starts(height, tile, step) for x in _starts(width, tile, step)]


def boxes_in_tile(boxes_xyxy, x0: int, y0: int, tile: int, min_visible: float):
    """(K, 4) frame pixels -> (K', 4) tile pixels: shift, clip, keep boxes >= min_visible inside."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4) - [x0, y0, x0, y0]
    clipped = np.clip(boxes, 0, tile)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    visible = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    return clipped[(visible > 0) & (visible >= min_visible * area)]


def _area(b):
    return (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])


def nms(dets, iou_threshold: float):
    """Greedy NMS on (K, 5) [x1, y1, x2, y2, score] -> kept rows, highest score first."""
    dets = np.asarray(dets, dtype=float).reshape(-1, 5)
    dets = dets[np.argsort(-dets[:, 4], kind="stable")]
    keep = []
    for i, d in enumerate(dets):
        kept = dets[keep]
        w = np.clip(np.minimum(d[2], kept[:, 2]) - np.maximum(d[0], kept[:, 0]), 0, None)
        h = np.clip(np.minimum(d[3], kept[:, 3]) - np.maximum(d[1], kept[:, 1]), 0, None)
        iou = w * h / np.maximum(_area(d) + _area(kept) - w * h, 1e-9)
        if not (iou > iou_threshold).any():
            keep.append(i)
    return dets[keep]


def predict_tiled(predict, frame, tile: int, overlap: float = 0.2, iou_threshold: float = 0.5):
    """SAHI inference on one frame with any detector.

    predict: list of images -> list of (K, 5) arrays in each image's own pixels.
    The full frame is predicted too (SAHI's default), so a large drone cut by tile borders
    is still found whole. Returns (K, 5) in frame pixels.
    """
    height, width = frame.shape[:2]
    origins = tile_origins(width, height, tile, overlap) + [(0, 0)]
    images = [frame[y : y + tile, x : x + tile] for x, y in origins[:-1]] + [frame]
    shifted = [
        np.asarray(d, dtype=float).reshape(-1, 5) + [x, y, x, y, 0]
        for (x, y), d in zip(origins, predict(images))
    ]
    return nms(np.concatenate(shifted), iou_threshold)
