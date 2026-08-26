"""Detector interface.

Detector is locked to YOLOX (CLAUDE.md), but every pipeline stage should only
ever depend on this interface — swapping the detector must be a config change
(pick a different InferenceBackend implementation), never a pipeline edit.
Real implementations (stage 4) wrap a TensorRT engine and own exactly one
execution context per worker thread, since TRT contexts aren't thread-safe
(the engine itself can be shared read-only across threads).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from streamguard.config import ModelTier, Precision


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class InferenceBackend(ABC):
    """Contract every detector backend (mock or real TensorRT) must satisfy."""

    @abstractmethod
    def load(self, tier: ModelTier, precision: Precision, input_size: int) -> None:
        """(Re)configure the backend for the given tier/precision/size. Must
        be safe to call again later to hot-swap tiers when the SLA controller
        moves along the degradation ladder."""

    @abstractmethod
    def infer(self, batch: np.ndarray) -> List[List[Detection]]:
        """Run inference on a batch of frames shaped (N, H, W, C) and return
        one detection list per frame, same order as the batch."""
