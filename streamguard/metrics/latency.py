"""Rolling-window latency tracking.

IMPORTANT: this module is deliberately clock-agnostic. It stores millisecond
values handed to it by `record()` and reports percentiles over them — it does
not measure anything itself. Per the project's measurement rules, the actual
timing of GPU stages (decode/preprocess/inference) MUST come from CUDA events,
never time.time()/time.perf_counter(), because host-side timing only captures
kernel *submission*, not completion, for async GPU work. That instrumentation
lives at the call site in the stages that do real GPU work (stage 3+); this
tracker only ever sees the resulting ms numbers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

STAGES = ("decode", "preprocess", "inference", "postprocess", "total")


@dataclass
class FrameLatency:
    """Per-stage latency breakdown for a single frame, in milliseconds."""

    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    stream_id: int = 0
    tier: int = 0
    timestamp: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "decode_ms": self.decode_ms,
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
            "total_ms": self.total_ms,
            "stream_id": self.stream_id,
            "tier": self.tier,
            "timestamp": self.timestamp,
        }


class LatencyTracker:
    """Bounded rolling window over `FrameLatency` records.

    The first `warmup_samples` records are counted but never enter the
    window, so cold-start effects (first-batch cache misses, lazy CUDA
    context init in later stages) don't pollute the percentiles the SLA
    controller will read.
    """

    def __init__(self, window_size: int = 200, warmup_samples: int = 30):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if warmup_samples < 0:
            raise ValueError("warmup_samples must be non-negative")
        self.window_size = window_size
        self.warmup_samples = warmup_samples
        self._window: Deque[FrameLatency] = deque(maxlen=window_size)
        self._seen = 0

    def record(self, sample: FrameLatency) -> None:
        self._seen += 1
        if self._seen <= self.warmup_samples:
            return
        self._window.append(sample)

    @property
    def is_warmed_up(self) -> bool:
        return self._seen > self.warmup_samples

    @property
    def samples(self) -> List[FrameLatency]:
        return list(self._window)

    @property
    def count(self) -> int:
        return len(self._window)

    @property
    def total_seen(self) -> int:
        return self._seen

    def _stage_array(self, stage: str) -> np.ndarray:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}, must be one of {STAGES}")
        attr = f"{stage}_ms"
        return np.array([getattr(s, attr) for s in self._window], dtype=np.float64)

    def percentile(self, p: float, stage: str = "total") -> Optional[float]:
        arr = self._stage_array(stage)
        if arr.size == 0:
            return None
        return float(np.percentile(arr, p))

    def p50(self, stage: str = "total") -> Optional[float]:
        return self.percentile(50, stage)

    def p95(self, stage: str = "total") -> Optional[float]:
        return self.percentile(95, stage)

    def p99(self, stage: str = "total") -> Optional[float]:
        return self.percentile(99, stage)

    def mean(self, stage: str = "total") -> Optional[float]:
        arr = self._stage_array(stage)
        if arr.size == 0:
            return None
        return float(np.mean(arr))

    def variance(self, stage: str = "total") -> Optional[float]:
        arr = self._stage_array(stage)
        if arr.size == 0:
            return None
        return float(np.var(arr))

    def stage_breakdown(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Median/p95/p99/variance for every stage in one call — this is the
        shape the results table / report should be built from."""
        return {
            stage: {
                "median": self.p50(stage),
                "p95": self.p95(stage),
                "p99": self.p99(stage),
                "variance": self.variance(stage),
            }
            for stage in STAGES
        }
