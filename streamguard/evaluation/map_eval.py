"""mAP evaluation: standard object-detection mean average precision,
computed against a labeled sample set at each degradation tier (CLAUDE.md's
"mAP measured at every degradation tier"). This is the harness only -- it
needs real predictions and real ground-truth annotations to mean anything;
there's no labeled data in this repo to run it against yet. Pure NumPy/
Python, so it's fully testable with synthetic boxes regardless.

AP is computed with all-point interpolation (the modern COCO/Pascal-VOC
convention): the precision-recall curve's envelope is made monotonically
non-increasing from the high-recall end, then the area under it is
integrated exactly at every point where recall changes. A useful
consequence, verified by a test below: a low-confidence duplicate
detection of an already-matched box counts as a false positive but does
NOT reduce AP if a higher-confidence detection already reached full recall
first -- that's a documented property of this metric, not a bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from streamguard.inference.base import Detection


@dataclass
class GroundTruthBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes_a: (Na, 4) x1,y1,x2,y2. boxes_b: (Nb, 4). Returns (Na, Nb) IoU."""
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated AP: area under the precision envelope
    (monotonically non-increasing from the right), integrated at every
    recall breakpoint."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match_single_class(
    predictions: List[Tuple[Any, float, Tuple[float, float, float, float]]],
    ground_truths: Dict[Any, List[Tuple[float, float, float, float]]],
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """predictions: (image_id, confidence, box), already sorted by
    confidence descending. Returns (tp, fp) as 0/1 float arrays, one entry
    per prediction, in the same order."""
    matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in ground_truths.items()}
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    for i, (img_id, _conf, box) in enumerate(predictions):
        gt_boxes = ground_truths.get(img_id)
        if not gt_boxes:
            fp[i] = 1
            continue
        ious = iou_matrix(np.array([box]), np.array(gt_boxes))[0]
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= iou_threshold and not matched[img_id][best_idx]:
            tp[i] = 1
            matched[img_id][best_idx] = True
        else:
            fp[i] = 1
    return tp, fp


def coco_iou_thresholds() -> List[float]:
    """COCO's mAP@[.5:.95]: 0.50, 0.55, ..., 0.95."""
    return [round(0.5 + 0.05 * i, 2) for i in range(10)]


def evaluate_detections(
    predictions: Dict[Any, List[Detection]],
    ground_truths: Dict[Any, List[GroundTruthBox]],
    num_classes: int,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """predictions/ground_truths keyed by the same image id. A class with
    no ground-truth instances at all is excluded from the mAP average at
    every threshold (the standard COCO convention), rather than counting
    as AP=0 and unfairly dragging the average down for a class that simply
    wasn't present in this sample."""
    iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5]
    image_ids = sorted(set(predictions.keys()) | set(ground_truths.keys()), key=str)

    per_class_ap: Dict[int, Dict[float, float]] = {}
    for class_id in range(num_classes):
        preds_this_class = sorted(
            (
                (img_id, d.confidence, (d.x1, d.y1, d.x2, d.y2))
                for img_id in image_ids
                for d in predictions.get(img_id, [])
                if d.class_id == class_id
            ),
            key=lambda p: p[1],
            reverse=True,
        )
        gts_this_class = {
            img_id: boxes
            for img_id in image_ids
            if (boxes := [(g.x1, g.y1, g.x2, g.y2) for g in ground_truths.get(img_id, []) if g.class_id == class_id])
        }
        num_gt = sum(len(v) for v in gts_this_class.values())
        if num_gt == 0:
            continue

        per_class_ap[class_id] = {}
        for threshold in iou_thresholds:
            if not preds_this_class:
                per_class_ap[class_id][threshold] = 0.0
                continue
            tp, fp = _match_single_class(preds_this_class, gts_this_class, threshold)
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recall = cum_tp / num_gt
            precision = cum_tp / np.clip(cum_tp + cum_fp, 1e-9, None)
            per_class_ap[class_id][threshold] = average_precision(recall, precision)

    map_per_threshold = {
        threshold: float(np.mean([per_class_ap[c][threshold] for c in per_class_ap]))
        for threshold in iou_thresholds
    } if per_class_ap else {threshold: 0.0 for threshold in iou_thresholds}

    overall_map = float(np.mean(list(map_per_threshold.values()))) if map_per_threshold else 0.0

    return {
        "per_class_ap": per_class_ap,
        "map_per_iou_threshold": map_per_threshold,
        "map": overall_map,
    }
