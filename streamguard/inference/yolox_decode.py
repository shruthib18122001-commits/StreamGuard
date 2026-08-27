"""Decodes YOLOX's raw network output into Detection objects.

YOLOX is anchor-free: predictions come from 3 feature-map strides (8, 16,
32), each contributing (input_size/stride)^2 grid-cell predictions. Per
grid cell, the raw tensor holds [reg_x, reg_y, reg_w, reg_h, obj,
cls_0..cls_{C-1}]. `reg` is a raw offset that needs decoding against the
grid position and stride; `obj`/`cls` arrive already sigmoid-activated,
because that's how YOLOX's own head builds its exported/inference-mode
output (`obj_output.sigmoid()` / `cls_output.sigmoid()` are concatenated
into the graph before export) -- nothing here needs an extra sigmoid.

This is pure NumPy math on tensor shapes: no GPU or TensorRT needed to
write or verify it, unlike the rest of stage 4. Output boxes stay in
letterboxed model-input coordinates, matching exactly what the mock
backend already returns, so postprocess.py's NMS/coordinate-inversion is
unchanged regardless of which backend produced the raw detections.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from streamguard.inference.base import Detection

YOLOX_STRIDES: Tuple[int, ...] = (8, 16, 32)


@lru_cache(maxsize=8)
def _grids_and_strides(input_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """For a square input_size, the (x, y) grid-cell coordinate and stride
    value for every anchor position, concatenated across all 3 feature-map
    strides in the same order the model emits them. Cached: input_size only
    changes on a degradation-tier switch, never per frame."""
    all_grids = []
    all_strides = []
    for stride in YOLOX_STRIDES:
        fm_size = input_size // stride
        yv, xv = np.meshgrid(np.arange(fm_size), np.arange(fm_size), indexing="ij")
        grid = np.stack((xv, yv), axis=-1).reshape(-1, 2)
        all_grids.append(grid)
        all_strides.append(np.full((grid.shape[0], 1), stride))
    grids = np.concatenate(all_grids, axis=0).astype(np.float64)
    strides = np.concatenate(all_strides, axis=0).astype(np.float64)
    return grids, strides


def decode_yolox_output(
    raw_output: np.ndarray,
    input_size: int,
    conf_threshold: float = 0.3,
) -> List[List[Detection]]:
    """raw_output: (batch, num_anchors, 5 + num_classes) float array, as
    produced by a YOLOX TensorRT engine. Returns one Detection list per
    frame in the batch, in letterboxed model-input coordinates -- NMS and
    mapping back to original-frame coordinates happen downstream in
    postprocess.py, same as for the mock backend."""
    if raw_output.ndim != 3:
        raise ValueError(
            f"expected raw_output shaped (batch, num_anchors, 5+num_classes), got shape {raw_output.shape}"
        )

    batch_size, num_anchors, channels = raw_output.shape
    num_classes = channels - 5
    if num_classes <= 0:
        raise ValueError(f"raw_output's last dim must be > 5, got {channels} (implies {num_classes} classes)")

    grids, strides = _grids_and_strides(input_size)
    if grids.shape[0] != num_anchors:
        raise ValueError(
            f"raw_output has {num_anchors} anchors but input_size={input_size} implies "
            f"{grids.shape[0]} -- wrong input_size for this output tensor?"
        )

    centers = (raw_output[..., 0:2] + grids) * strides  # (batch, num_anchors, 2)
    sizes = np.exp(raw_output[..., 2:4]) * strides  # (batch, num_anchors, 2)
    obj_scores = raw_output[..., 4]  # (batch, num_anchors)
    cls_scores = raw_output[..., 5:]  # (batch, num_anchors, num_classes)

    x1y1 = centers - sizes / 2.0
    x2y2 = centers + sizes / 2.0

    best_cls_id = np.argmax(cls_scores, axis=-1)  # (batch, num_anchors)
    best_cls_score = np.take_along_axis(cls_scores, best_cls_id[..., None], axis=-1)[..., 0]
    confidence = obj_scores * best_cls_score  # (batch, num_anchors)

    results: List[List[Detection]] = []
    for b in range(batch_size):
        keep = confidence[b] >= conf_threshold
        if not np.any(keep):
            results.append([])
            continue
        boxes = np.concatenate([x1y1[b, keep], x2y2[b, keep]], axis=-1)
        scores = confidence[b, keep]
        class_ids = best_cls_id[b, keep]
        results.append(
            [
                Detection(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]),
                          confidence=float(s), class_id=int(c))
                for box, s, c in zip(boxes, scores, class_ids)
            ]
        )
    return results
