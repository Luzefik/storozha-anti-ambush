"""The one evaluator every model is scored with.

Settings (IoU thresholds, confidence threshold, max detections) come from
config/default.yaml `eval:` and are identical for every model.
"""

from __future__ import annotations

import numpy as np


def evaluate(
    preds: dict[str, list[np.ndarray]],
    gts: dict[str, list[np.ndarray]],
    iou_thresholds: list[float],
) -> dict:
    """
    preds: video -> per-frame arrays (K_i, 5) [x1, y1, x2, y2, score]
    gts:   video -> per-frame arrays (M_i, 4) [x1, y1, x2, y2]  (M_i may be 0)
    returns: metrics overall AND per video, so day/night and per-category breakdowns
             and video-level bootstrap CIs can be computed from the same output.
    """
    # TODO: Implement (or wrap pycocotools / torchmetrics). Keep per-video results:
    # the resampling unit for confidence intervals is the video, not the frame.
    raise NotImplementedError
