"""Stream-count sweep: ramp concurrent stream count at a fixed degradation
tier until p99 violates the SLA deadline, to find the max sustainable N for
that tier (CLAUDE.md's "sweep stream count upward until SLA violation").

Runs against any InferenceBackend -- MockInferenceBackend today,
TensorRTInferenceBackend once stage 4 is verified on real hardware. With
the mock backend the resulting "sustainable N" numbers are placeholders,
not real capacity figures, but the sweep logic itself -- ramping N,
checking the deadline, stopping at the first violation, recording results
-- is real and is exactly how the real sweep will run later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from streamguard.config import BatchConfig, DegradationTier, QueueConfig
from streamguard.decode.base import Decoder
from streamguard.inference.base import InferenceBackend
from streamguard.metrics.latency import LatencyTracker
from streamguard.multi_stream_pipeline import MultiStreamPipeline
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.base import Preprocessor


@dataclass
class StreamCountTrialResult:
    stream_count: int
    tier_level: int
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]
    frames_processed: int
    dropped_decode_queue: int
    dropped_batch_queue: int
    sla_violated: bool


def run_stream_count_trial(
    stream_count: int,
    tier: DegradationTier,
    sources: List[str],
    decoder_factory: Callable[[], Decoder],
    preprocessor: Preprocessor,
    inference_backend: InferenceBackend,
    deadline_ms: float,
    queue_config: Optional[QueueConfig] = None,
    batch_config: Optional[BatchConfig] = None,
    max_frames_per_stream: Optional[int] = None,
    warmup_samples: int = 10,
    join_timeout: float = 60.0,
) -> StreamCountTrialResult:
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
    result = pipeline.run(trial_sources, max_frames_per_stream=max_frames_per_stream, join_timeout=join_timeout)

    p99 = tracker.p99()
    return StreamCountTrialResult(
        stream_count=stream_count,
        tier_level=tier.level,
        p50_ms=tracker.p50(),
        p95_ms=tracker.p95(),
        p99_ms=p99,
        frames_processed=result["frames_processed"],
        dropped_decode_queue=result["dropped_decode_queue"],
        dropped_batch_queue=result["dropped_batch_queue"],
        sla_violated=(p99 is not None and p99 > deadline_ms),
    )


def sweep_stream_counts(
    stream_counts: List[int],
    tier: DegradationTier,
    sources: List[str],
    decoder_factory: Callable[[], Decoder],
    preprocessor: Preprocessor,
    inference_backend: InferenceBackend,
    deadline_ms: float,
    **trial_kwargs,
) -> List[StreamCountTrialResult]:
    """Runs increasing stream counts, stopping one trial past the first SLA
    violation -- enough to show the falloff, without wasting time
    continuing to climb once it's already broken."""
    results: List[StreamCountTrialResult] = []
    for n in stream_counts:
        trial = run_stream_count_trial(
            n, tier, sources, decoder_factory, preprocessor, inference_backend, deadline_ms, **trial_kwargs
        )
        results.append(trial)
        if trial.sla_violated:
            break
    return results


def max_sustainable_stream_count(results: List[StreamCountTrialResult]) -> Optional[int]:
    passing = [r.stream_count for r in results if not r.sla_violated]
    return max(passing) if passing else None
