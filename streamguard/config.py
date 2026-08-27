"""Config schema and YAML loader for StreamGuard.

Pydantic models, not plain dataclasses: we want the same fail-fast validation
you'd get from a FastAPI request body, applied to the experiment config instead.
A bad threshold or a negative stream count should error at load time, not three
stages deep into a Colab run.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelTier(str, Enum):
    NANO = "nano"
    TINY = "tiny"
    S = "s"
    M = "m"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class ModelConfig(BaseModel):
    """Baseline (tier 0) model selection. Detector is YOLOX; swapping detectors
    should only ever mean pointing this at a different InferenceBackend impl,
    never editing pipeline code."""

    tier: ModelTier = ModelTier.M
    precision: Precision = Precision.FP16
    input_size: int = 640
    weights_path: Optional[str] = None

    @field_validator("input_size")
    @classmethod
    def _positive_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("input_size must be positive")
        return v


class DegradationTier(BaseModel):
    """One rung of the degradation ladder. `level` 0 is the highest-quality,
    highest-cost rung; higher levels sacrifice more to protect the deadline."""

    level: int
    model_tier: ModelTier
    precision: Precision
    input_size: int
    frame_skip: int = Field(default=1, ge=1)
    description: str = ""


DEFAULT_LADDER: List[DegradationTier] = [
    DegradationTier(level=0, model_tier=ModelTier.M, precision=Precision.FP16, input_size=640, description="baseline"),
    DegradationTier(level=1, model_tier=ModelTier.S, precision=Precision.FP16, input_size=640, description="drop to YOLOX-s"),
    DegradationTier(level=2, model_tier=ModelTier.S, precision=Precision.INT8, input_size=640, description="INT8 quantize"),
    DegradationTier(level=3, model_tier=ModelTier.TINY, precision=Precision.INT8, input_size=416, description="YOLOX-tiny @ 416px"),
    DegradationTier(level=4, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416, description="YOLOX-nano"),
    DegradationTier(level=5, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416, frame_skip=2, description="selective frame skip"),
]


class SLAConfig(BaseModel):
    """Deadline and hysteresis parameters the (stage 5) SLA controller will
    read. Defined now so the schema is stable; the controller that walks
    `ladder` using these thresholds doesn't exist yet."""

    deadline_ms: float = 33.0
    degrade_threshold_ms: float = 30.0
    recover_threshold_ms: float = 20.0
    min_dwell_time_s: float = 2.0
    window_size: int = Field(default=200, gt=0)
    warmup_samples: int = Field(default=30, ge=0)
    ladder: List[DegradationTier] = Field(default_factory=lambda: list(DEFAULT_LADDER))

    @model_validator(mode="after")
    def _hysteresis_sane(self) -> "SLAConfig":
        if self.recover_threshold_ms >= self.degrade_threshold_ms:
            raise ValueError(
                "recover_threshold_ms must be < degrade_threshold_ms — "
                "equal or inverted thresholds collapse the hysteresis band "
                "and the controller will oscillate every window."
            )
        if self.degrade_threshold_ms <= 0 or self.recover_threshold_ms <= 0:
            raise ValueError("SLA thresholds must be positive")
        return self

    @model_validator(mode="after")
    def _ladder_sane(self) -> "SLAConfig":
        levels = [t.level for t in self.ladder]
        if levels != sorted(levels) or len(set(levels)) != len(levels):
            raise ValueError("ladder levels must be strictly increasing, no duplicates")
        return self


class StreamConfig(BaseModel):
    count: int = Field(default=1, gt=0)
    sources: Optional[List[str]] = None


class BackpressurePolicy(str, Enum):
    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"


class QueueConfig(BaseModel):
    """Bounded-queue sizes and backpressure policy for each hop in the
    pipeline. Pre-inference queues default to dropping the oldest frame when
    full — a stale frame only burns deadline budget, so skipping ahead to a
    fresher one is strictly better. The post-inference queue defaults to
    blocking instead, since by then real (simulated) GPU time has already
    been spent on that frame; throwing away a completed result is pure
    waste that a blocked producer upstream doesn't cost us."""

    decode_queue_size: int = Field(default=8, gt=0)
    batch_queue_size: int = Field(default=16, gt=0)
    postprocess_queue_size: int = Field(default=32, gt=0)
    decode_queue_policy: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST
    batch_queue_policy: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST
    postprocess_queue_policy: BackpressurePolicy = BackpressurePolicy.BLOCK


class BatchConfig(BaseModel):
    """Dynamic batching: accumulate frames from any stream until either
    `max_batch_size` is reached or `batch_timeout_ms` has elapsed since the
    first frame of the current batch arrived, whichever fires first."""

    max_batch_size: int = Field(default=8, gt=0)
    batch_timeout_ms: float = Field(default=10.0, gt=0)


class StreamGuardConfig(BaseModel):
    streams: StreamConfig = Field(default_factory=StreamConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sla: SLAConfig = Field(default_factory=SLAConfig)
    queues: QueueConfig = Field(default_factory=QueueConfig)
    batching: BatchConfig = Field(default_factory=BatchConfig)
    device: str = "cuda:0"


def load_config(path: Union[str, Path]) -> StreamGuardConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return StreamGuardConfig(**raw)
