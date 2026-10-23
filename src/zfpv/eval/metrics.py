"""The one evaluator every model is scored with.

Settings (IoU thresholds, confidence threshold, max detections) come from
config/default.yaml `eval:` and are identical for every model.

One class. COCO-style matching: per frame, predictions in score order each take the
unmatched ground-truth box with the highest IoU; a hit needs IoU >= threshold, and a second
box on the same drone is a false positive. AP = COCO 101-point interpolation.
Results come overall, per video and per group (day / night): the video is the unit.
"""

from __future__ import annotations

import numpy as np


def _area(b):
    return (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])


def iou_matrix(a, b) -> np.ndarray:
    """IoU between boxes a (N, 4) and b (M, 4), [x1, y1, x2, y2] -> (N, M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 1, 4)
    b = np.asarray(b, dtype=float).reshape(1, -1, 4)
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    return w * h / np.maximum(_area(a) + _area(b) - w * h, 1e-12)


def match_frame(pred, gt, iou_threshold: float, max_detections: int | None = None):
    """One frame -> (scores, is_tp) of its predictions, highest score first."""
    pred = np.asarray(pred, dtype=float).reshape(-1, 5)
    pred = pred[np.argsort(-pred[:, 4], kind="stable")][:max_detections]
    ious = iou_matrix(pred[:, :4], gt)
    is_tp = np.zeros(len(pred), dtype=bool)
    for i in range(len(pred)):
        if ious.shape[1] and ious[i].max() >= iou_threshold:
            j = ious[i].argmax()
            is_tp[i], ious[:, j] = True, -1.0  # this drone is taken
    return pred[:, 4], is_tp


def average_precision(scores, is_tp, n_gt: int) -> float:
    """COCO 101-point interpolated AP; NaN when there is no ground truth."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    tp = np.cumsum(np.asarray(is_tp, dtype=bool)[order])
    recall, precision = tp / n_gt, tp / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]  # best precision at recall >= r
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(tp), envelope[np.minimum(idx, len(tp) - 1)], 0).mean())


def evaluate(
    preds: dict[str, list[np.ndarray]],
    gts: dict[str, list[np.ndarray]],
    iou_thresholds: list[float],
    conf_threshold: float | None = None,
    max_detections: int | None = 100,
    groups: dict[str, str] | None = None,
) -> dict:
    """
    preds: video -> per-frame arrays (K_i, 5) [x1, y1, x2, y2, score]
    gts:   video -> per-frame arrays (M_i, 4) [x1, y1, x2, y2]  (M_i may be 0)
    conf_threshold: operating point for precision / recall (None = the best-F1 score).
    groups: video -> group name, e.g. "day" / "night".
    returns {"overall", "per_video", "per_group"}, each with n_frames, n_gt, AP@<thr>, mAP,
    and conf / precision / recall @<first thr>.
    """
    if set(preds) != set(gts) or any(len(preds[v]) != len(gts[v]) for v in gts):
        raise ValueError("preds and gts must have the same videos and frame counts")
    # match once per video and threshold, then pool for any set of videos
    matched = {
        (v, t): [match_frame(p, g, t, max_detections) for p, g in zip(preds[v], gts[v])]
        for v in gts
        for t in iou_thresholds
    }

    def summary(videos: list[str]) -> dict:
        n_gt = sum(len(np.reshape(g, (-1, 4))) for v in videos for g in gts[v])
        out = {"n_frames": sum(len(gts[v]) for v in videos), "n_gt": n_gt}
        for t in iou_thresholds:
            pairs = [m for v in videos for m in matched[v, t]]
            scores = np.concatenate([s for s, _ in pairs]) if pairs else np.zeros(0)
            is_tp = np.concatenate([k for _, k in pairs]) if pairs else np.zeros(0, bool)
            out[f"AP@{t:g}"] = average_precision(scores, is_tp, n_gt)
            if t == iou_thresholds[0]:
                op = operating_point(scores, is_tp, n_gt, conf_threshold)
                out.update({f"{k}@{t:g}": val for k, val in op.items()})
        out["mAP"] = float(np.mean([out[f"AP@{t:g}"] for t in iou_thresholds]))
        return out

    videos = sorted(gts)
    by_group: dict[str, list[str]] = {}
    for v in videos:
        by_group.setdefault((groups or {}).get(v, "all"), []).append(v)
    return {
        "overall": summary(videos),
        "per_video": {v: summary([v]) for v in videos},
        "per_group": {g: summary(vs) for g, vs in sorted(by_group.items())} if groups else {},
    }


def operating_point(scores, is_tp, n_gt: int, conf_threshold: float | None) -> dict:
    """Precision / recall at conf_threshold, or at the best-F1 score when it is None."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, tp = scores[order], np.cumsum(np.asarray(is_tp, bool)[order])
    if conf_threshold is not None:
        keep = s >= conf_threshold
        s, tp = s[keep], tp[keep]
    if len(s) == 0 or n_gt == 0:
        return {"conf": conf_threshold, "precision": float("nan"), "recall": 0.0}
    precision, recall = tp / np.arange(1, len(tp) + 1), tp / n_gt
    if conf_threshold is None:
        k = int(np.argmax(2 * precision * recall / np.maximum(precision + recall, 1e-12)))
    else:
        k = len(s) - 1
    conf = float(s[k]) if conf_threshold is None else conf_threshold
    return {"conf": conf, "precision": float(precision[k]), "recall": float(recall[k])}
