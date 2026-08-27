import numpy as np
import pytest

from streamguard.evaluation.map_eval import (
    GroundTruthBox,
    average_precision,
    coco_iou_thresholds,
    evaluate_detections,
    iou_matrix,
)
from streamguard.inference.base import Detection


def test_iou_matrix_identical_boxes_is_one():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(boxes, boxes)[0, 0] == pytest.approx(1.0)


def test_iou_matrix_disjoint_boxes_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_average_precision_perfect_detector_is_one():
    # Every prediction correct, recall reaches 1.0 immediately.
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert average_precision(recall, precision) == pytest.approx(1.0)


def test_average_precision_no_correct_predictions_is_zero():
    recall = np.array([0.0, 0.0])
    precision = np.array([0.0, 0.0])
    assert average_precision(recall, precision) == pytest.approx(0.0)


def test_evaluate_single_perfect_match():
    predictions = {"img1": [Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0)]}
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}

    result = evaluate_detections(predictions, ground_truths, num_classes=1, iou_thresholds=[0.5])

    assert result["map"] == pytest.approx(1.0)
    assert result["per_class_ap"][0][0.5] == pytest.approx(1.0)


def test_evaluate_zero_overlap_prediction_is_zero_ap():
    predictions = {"img1": [Detection(x1=100, y1=100, x2=110, y2=110, confidence=0.9, class_id=0)]}
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}

    result = evaluate_detections(predictions, ground_truths, num_classes=1, iou_thresholds=[0.5])

    assert result["map"] == pytest.approx(0.0)


def test_evaluate_duplicate_detection_does_not_reduce_ap():
    # A high-confidence correct box plus a lower-confidence duplicate of the
    # SAME ground truth: the duplicate is a false positive (can't match a
    # box that's already claimed), but since recall already hit 1.0 at the
    # higher-confidence detection, AP stays 1.0 -- a documented property of
    # all-point interpolation, not a bug.
    predictions = {
        "img1": [
            Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0),
            Detection(x1=0.5, y1=0.5, x2=10.5, y2=10.5, confidence=0.4, class_id=0),
        ]
    }
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}

    result = evaluate_detections(predictions, ground_truths, num_classes=1, iou_thresholds=[0.5])

    assert result["per_class_ap"][0][0.5] == pytest.approx(1.0)


def test_evaluate_missed_detection_gives_zero_ap_for_that_class():
    predictions = {"img1": []}
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}

    result = evaluate_detections(predictions, ground_truths, num_classes=1, iou_thresholds=[0.5])

    assert result["per_class_ap"][0][0.5] == pytest.approx(0.0)


def test_class_absent_from_ground_truth_excluded_from_map():
    # class 1 never appears in ground truth at all -- must be excluded from
    # the average, not counted as AP=0 and drag mAP down unfairly.
    predictions = {"img1": [Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0)]}
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}

    result = evaluate_detections(predictions, ground_truths, num_classes=2, iou_thresholds=[0.5])

    assert 1 not in result["per_class_ap"]
    assert result["map"] == pytest.approx(1.0)


def test_multiple_iou_thresholds_perfect_match_all_ones():
    predictions = {"img1": [Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0)]}
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}

    result = evaluate_detections(predictions, ground_truths, num_classes=1, iou_thresholds=coco_iou_thresholds())

    assert result["map"] == pytest.approx(1.0)
    assert len(result["map_per_iou_threshold"]) == 10


def test_coco_iou_thresholds_span_0_5_to_0_95():
    thresholds = coco_iou_thresholds()
    assert thresholds[0] == pytest.approx(0.5)
    assert thresholds[-1] == pytest.approx(0.95)
    assert len(thresholds) == 10
