"""Postprocessing: confidence filter, NMS, then map surviving boxes from
letterboxed model-input space back to the original frame's coordinates.

No per-detector backend split like Decoder/Preprocessor/InferenceBackend —
this stays vectorized NumPy regardless of whether inference ran on GPU or is
mocked (per CLAUDE.md: "postprocessing stays on GPU or vectorized NumPy").
"""
from __future__ import annotations

from typing import List

import numpy as np

from streamguard.inference.base import Detection
from streamguard.preprocess.base import PreprocessMeta


class Postprocessor:
    def __init__(self, conf_threshold: float = 0.3, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def process(self, detections: List[Detection], meta: PreprocessMeta) -> List[Detection]:
        if not detections:
            return []

        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float64)
        scores = np.array([d.confidence for d in detections], dtype=np.float64)
        class_ids = np.array([d.class_id for d in detections], dtype=np.int64)

        keep_conf = scores >= self.conf_threshold
        boxes, scores, class_ids = boxes[keep_conf], scores[keep_conf], class_ids[keep_conf]
        if boxes.shape[0] == 0:
            return []

        keep_idx = self._nms(boxes, scores, self.iou_threshold)
        boxes, scores, class_ids = boxes[keep_idx], scores[keep_idx], class_ids[keep_idx]

        # Invert the letterbox transform: undo padding, then undo scale.
        boxes[:, [0, 2]] -= meta.pad_x
        boxes[:, [1, 3]] -= meta.pad_y
        boxes /= meta.scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta.orig_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta.orig_height)

        return [
            Detection(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
                      confidence=float(s), class_id=int(c))
            for b, s, c in zip(boxes, scores, class_ids)
        ]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Vectorized greedy NMS — O(k^2) per frame over a handful of boxes,
        not a per-box Python-level loop over pixels, so it's fine on CPU."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_threshold]

        return np.array(keep, dtype=np.int64)
