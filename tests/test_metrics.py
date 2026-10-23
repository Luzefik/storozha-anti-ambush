"""The shared evaluator on hand-made boxes: no gated data, runs in CI."""

import math

import numpy as np
import pytest

from zfpv.eval.metrics import evaluate, iou_matrix, match_frame

GT = np.array([[100, 100, 120, 120]], dtype=float)
NONE = np.zeros((0, 4))
NO_PRED = np.zeros((0, 5))


def det(box, score):
    return np.array([[*box, score]], dtype=float)


def test_iou_of_identical_partial_and_disjoint_boxes():
    ious = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(ious, [[1.0, 1 / 3, 0.0]])


def test_second_box_on_the_same_drone_is_a_false_positive():
    pred = np.array([[100, 100, 120, 120, 0.9], [101, 101, 121, 121, 0.8]])
    assert match_frame(pred, GT, 0.5)[1].tolist() == [True, False]


def test_iou_threshold_decides_a_hit():
    shifted = det([105, 100, 125, 120], 0.9)  # IoU 0.6
    assert match_frame(shifted, GT, 0.5)[1].tolist() == [True]
    assert match_frame(shifted, GT, 0.75)[1].tolist() == [False]


def test_perfect_detector_scores_one():
    r = evaluate({"v": [det(GT[0], 0.9), NO_PRED]}, {"v": [GT, NONE]}, [0.5, 0.75])
    assert r["overall"]["AP@0.5"] == pytest.approx(1.0)
    assert r["overall"]["mAP"] == pytest.approx(1.0)
    assert r["overall"]["recall@0.5"] == pytest.approx(1.0)


def test_no_predictions_score_zero():
    r = evaluate({"v": [NO_PRED, NO_PRED]}, {"v": [GT, GT]}, [0.5])
    assert r["overall"]["AP@0.5"] == 0.0 and r["overall"]["recall@0.5"] == 0.0


def test_only_false_alarms_ranked_above_the_drone_cost_ap():
    gts = {"v": [GT, NONE]}
    high_fp = {"v": [det(GT[0], 0.5), det([300, 300, 320, 320], 0.9)]}
    low_fp = {"v": [det(GT[0], 0.9), det([300, 300, 320, 320], 0.5)]}
    assert evaluate(high_fp, gts, [0.5])["overall"]["AP@0.5"] == pytest.approx(0.5)
    assert evaluate(low_fp, gts, [0.5])["overall"]["AP@0.5"] == pytest.approx(1.0)


def test_conf_threshold_operating_point():
    r = evaluate({"v": [det(GT[0], 0.9), det(GT[0], 0.2)]}, {"v": [GT, GT]}, [0.5], 0.5)
    assert r["overall"]["recall@0.5"] == pytest.approx(0.5)
    assert r["overall"]["precision@0.5"] == pytest.approx(1.0)


def test_per_video_and_day_night_groups():
    preds = {"d": [det(GT[0], 0.9)], "n": [NO_PRED]}
    r = evaluate(preds, {"d": [GT], "n": [GT]}, [0.5], groups={"d": "day", "n": "night"})
    assert r["per_video"]["d"]["AP@0.5"] == pytest.approx(1.0)
    assert r["per_group"]["night"]["recall@0.5"] == 0.0
    assert r["overall"]["recall@0.5"] == pytest.approx(0.5)


def test_video_without_drones_has_nan_ap():
    assert math.isnan(evaluate({"v": [NO_PRED]}, {"v": [NONE]}, [0.5])["overall"]["AP@0.5"])


def test_mismatched_frames_raise():
    with pytest.raises(ValueError):
        evaluate({"v": [NO_PRED]}, {"v": [GT, GT]}, [0.5])
