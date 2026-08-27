"""mAP-per-tier evaluation: an offline pass over a small labeled sample
set, run at each degradation tier, feeding streamguard.evaluation.map_eval.
Deliberately bypasses the queueing/threading pipeline entirely -- mAP
measurement is "run the detector over a fixed labeled set and compare to
ground truth," not a live-streaming concern, so it doesn't need
MultiStreamPipeline's machinery at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from streamguard.config import DegradationTier
from streamguard.evaluation.map_eval import GroundTruthBox, evaluate_detections
from streamguard.inference.base import Detection, InferenceBackend
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.base import Preprocessor


def run_map_trial(
    tier: DegradationTier,
    preprocessor: Preprocessor,
    inference_backend: InferenceBackend,
    postprocessor: Postprocessor,
    labeled_frames: Dict[Any, np.ndarray],
    ground_truths: Dict[Any, List[GroundTruthBox]],
    num_classes: int,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """labeled_frames: image_id -> raw (H, W, 3) frame. Runs every frame
    through this tier's configuration and returns evaluate_detections()'s
    result dict (per-class AP, mAP per IoU threshold, overall mAP)."""
    inference_backend.load(tier.model_tier, tier.precision, tier.input_size)

    predictions: Dict[Any, List[Detection]] = {}
    for image_id, frame in labeled_frames.items():
        processed, meta = preprocessor.process(frame, tier.input_size)
        batch = processed[np.newaxis, ...]
        raw_detections = inference_backend.infer(batch)
        predictions[image_id] = postprocessor.process(raw_detections[0], meta)

    return evaluate_detections(predictions, ground_truths, num_classes, iou_thresholds)


def sweep_map_per_tier(
    ladder: List[DegradationTier],
    preprocessor: Preprocessor,
    inference_backend: InferenceBackend,
    postprocessor: Postprocessor,
    labeled_frames: Dict[Any, np.ndarray],
    ground_truths: Dict[Any, List[GroundTruthBox]],
    num_classes: int,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[int, Dict[str, Any]]:
    """One evaluate_detections() result per ladder rung, keyed by tier
    level."""
    return {
        tier.level: run_map_trial(
            tier, preprocessor, inference_backend, postprocessor,
            labeled_frames, ground_truths, num_classes, iou_thresholds,
        )
        for tier in ladder
    }
