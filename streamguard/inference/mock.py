"""Mock inference backend.

Simulates a detector's per-batch latency (via time.sleep — this is a stand-in
CPU-side delay for a GPU that doesn't exist yet, not a measurement of real GPU
work, so it doesn't run afoul of the CUDA-events-only rule) and returns
random-but-shaped detections. Exists so the queue/batching engine and the SLA
controller (stages 3 and 5) can be exercised end-to-end, including tier
hot-swaps under simulated load, entirely on CPU.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from streamguard.config import ModelTier, Precision
from streamguard.inference.base import Detection, InferenceBackend

# Placeholder relative costs, monotonic within a tier by precision and across
# tiers by size — good enough to exercise degradation-ladder logic. Not real
# measurements; those come from stage 4's actual TensorRT engines.
DEFAULT_TIER_LATENCY_MS: Dict[Tuple[ModelTier, Precision], float] = {
    (ModelTier.M, Precision.FP32): 45.0,
    (ModelTier.M, Precision.FP16): 25.0,
    (ModelTier.M, Precision.INT8): 18.0,
    (ModelTier.S, Precision.FP32): 30.0,
    (ModelTier.S, Precision.FP16): 15.0,
    (ModelTier.S, Precision.INT8): 10.0,
    (ModelTier.TINY, Precision.FP32): 18.0,
    (ModelTier.TINY, Precision.FP16): 9.0,
    (ModelTier.TINY, Precision.INT8): 6.0,
    (ModelTier.NANO, Precision.FP32): 10.0,
    (ModelTier.NANO, Precision.FP16): 5.0,
    (ModelTier.NANO, Precision.INT8): 3.0,
}


class MockInferenceBackend(InferenceBackend):
    def __init__(
        self,
        latency_table: Optional[Dict[Tuple[ModelTier, Precision], float]] = None,
        jitter_ms: float = 0.0,
        max_detections_per_frame: int = 3,
        rng_seed: Optional[int] = None,
    ):
        self._latency_table = DEFAULT_TIER_LATENCY_MS if latency_table is None else latency_table
        self._jitter_ms = jitter_ms
        self._max_detections = max_detections_per_frame
        self._rng = np.random.default_rng(rng_seed)
        self._tier: Optional[ModelTier] = None
        self._precision: Optional[Precision] = None
        self._input_size: Optional[int] = None
        self._loaded = False

    def load(self, tier: ModelTier, precision: Precision, input_size: int) -> None:
        if (tier, precision) not in self._latency_table:
            raise ValueError(f"no mock latency configured for {tier}/{precision}")
        self._tier = tier
        self._precision = precision
        self._input_size = input_size
        self._loaded = True

    def infer(self, batch: np.ndarray) -> List[List[Detection]]:
        if not self._loaded:
            raise RuntimeError("MockInferenceBackend.infer() called before load()")
        n = batch.shape[0]
        base_ms = self._latency_table[(self._tier, self._precision)]
        jitter = float(self._rng.uniform(-self._jitter_ms, self._jitter_ms)) if self._jitter_ms else 0.0
        delay_s = max(0.0, base_ms + jitter) / 1000.0
        time.sleep(delay_s)
        return [self._fake_detections() for _ in range(n)]

    def _fake_detections(self) -> List[Detection]:
        n_dets = int(self._rng.integers(0, self._max_detections + 1))
        detections = []
        for _ in range(n_dets):
            x1, y1 = self._rng.uniform(0, self._input_size, size=2)
            w, h = self._rng.uniform(5, max(6.0, self._input_size / 4), size=2)
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x1 + w),
                    y2=float(y1 + h),
                    confidence=float(self._rng.uniform(0.3, 0.99)),
                    class_id=int(self._rng.integers(0, 80)),
                )
            )
        return detections

    @property
    def current_config(self) -> Dict[str, object]:
        return {"tier": self._tier, "precision": self._precision, "input_size": self._input_size}
