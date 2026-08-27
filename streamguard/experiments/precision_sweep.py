"""Precision sweep: run each rung of the degradation ladder at a fixed
stream count and record latency/throughput/power (CLAUDE.md's "Precision
sweep: FP32/FP16/INT8 on latency, throughput, power, mAP" -- mAP is a
separate, offline concern; see map_trial.py). Power comes from an
NVMLSampler bracketing each trial, when one's supplied and available.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from streamguard.config import BatchConfig, DegradationTier, QueueConfig
from streamguard.decode.base import Decoder
from streamguard.inference.base import InferenceBackend
from streamguard.metrics.latency import LatencyTracker
from streamguard.metrics.nvml_sampler import NVMLSampler
from streamguard.multi_stream_pipeline import MultiStreamPipeline
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.base import Preprocessor


@dataclass
class PrecisionTrialResult:
    tier_level: int
    model_tier: str
    precision: str
    input_size: int
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]
    throughput_fps: Optional[float]
    frames_processed: int
    mean_power_watts: Optional[float]


def run_precision_trial(
    tier: DegradationTier,
    stream_count: int,
    sources: List[str],
    decoder_factory: Callable[[], Decoder],
    preprocessor: Preprocessor,
    inference_backend: InferenceBackend,
    queue_config: Optional[QueueConfig] = None,
    batch_config: Optional[BatchConfig] = None,
    max_frames_per_stream: Optional[int] = None,
    warmup_samples: int = 10,
    join_timeout: float = 60.0,
    nvml_sampler: Optional[NVMLSampler] = None,
) -> PrecisionTrialResult:
    trial_sources = [sources[i % len(sources)] for i in range(stream_count)]
    inference_backend.load(tier.model_tier, tier.precision, tier.input_size)
    tracker = LatencyTracker(window_size=max(200, stream_count * 20), warmup_samples=warmup_samples)
    pipeline = MultiStreamPipeline(
        decoder_factory=decoder_factory,
        preprocessor=preprocessor,
        inference_backend=inference_backend,
        postprocessor=Postprocessor(),
        tracker=tracker,
        model_input_size=tier.input_size,
        queue_config=queue_config or QueueConfig(),
        batch_config=batch_config or BatchConfig(),
    )

    if nvml_sampler is not None:
        nvml_sampler.start()
    start = time.perf_counter()
    result = pipeline.run(trial_sources, max_frames_per_stream=max_frames_per_stream, join_timeout=join_timeout)
    wall_s = time.perf_counter() - start
    if nvml_sampler is not None:
        nvml_sampler.stop()

    mean_power = None
    if nvml_sampler is not None and nvml_sampler.samples:
        powers = [s.power_watts for s in nvml_sampler.samples if s.power_watts is not None]
        mean_power = sum(powers) / len(powers) if powers else None

    return PrecisionTrialResult(
        tier_level=tier.level,
        model_tier=tier.model_tier.value,
        precision=tier.precision.value,
        input_size=tier.input_size,
        p50_ms=tracker.p50(),
        p95_ms=tracker.p95(),
        p99_ms=tracker.p99(),
        throughput_fps=(result["frames_processed"] / wall_s) if wall_s > 0 else None,
        frames_processed=result["frames_processed"],
        mean_power_watts=mean_power,
    )


def sweep_precision(
    ladder: List[DegradationTier],
    stream_count: int,
    sources: List[str],
    decoder_factory: Callable[[], Decoder],
    preprocessor: Preprocessor,
    inference_backend: InferenceBackend,
    **trial_kwargs,
) -> List[PrecisionTrialResult]:
    """One trial per rung of the ladder, in order."""
    return [
        run_precision_trial(
            tier, stream_count, sources, decoder_factory, preprocessor, inference_backend, **trial_kwargs
        )
        for tier in ladder
    ]
