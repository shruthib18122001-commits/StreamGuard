#!/usr/bin/env python3
"""Demo: watch the SLA controller actually degrade and reload the model
under real concurrent load, then print the resulting tier-transition log.

No GPU needed -- MockInferenceBackend simulates a baseline tier slow
enough to violate the 33ms deadline under 4 concurrent streams, a middle
tier that's faster, and a cheap tier that's comfortably under. This is the
core mechanism the whole project exists to demonstrate; everything else
(queues, batching, TensorRT) is in service of this working correctly.
"""
from __future__ import annotations

from streamguard.config import (
    BatchConfig,
    DegradationTier,
    ModelTier,
    Precision,
    QueueConfig,
    SLAConfig,
)
from streamguard.decode.cpu import CPUDecoder
from streamguard.experiments.sources import generate_synthetic_sources
from streamguard.inference.mock import MockInferenceBackend
from streamguard.metrics.latency import LatencyTracker
from streamguard.multi_stream_pipeline import MultiStreamPipeline
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor
from streamguard.sla_controller import SLAController

# A 3-tier ladder: baseline slow enough to violate 33ms under 4 concurrent
# streams, a middle tier that recovers it, a cheap tier that's comfortable.
DEMO_LADDER = [
    DegradationTier(level=0, model_tier=ModelTier.M, precision=Precision.FP16, input_size=640, description="baseline"),
    DegradationTier(level=1, model_tier=ModelTier.S, precision=Precision.FP16, input_size=640, description="drop to s/fp16"),
    DegradationTier(level=2, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416, description="nano/int8"),
]

DEMO_LATENCY_TABLE = {
    (ModelTier.M, Precision.FP16): 40.0,
    (ModelTier.S, Precision.FP16): 20.0,
    (ModelTier.NANO, Precision.INT8): 5.0,
}


def main() -> None:
    backend = MockInferenceBackend(latency_table=DEMO_LATENCY_TABLE, jitter_ms=0.0)
    backend.load(ModelTier.M, Precision.FP16, 640)

    controller = SLAController(
        SLAConfig(degrade_threshold_ms=33.0, recover_threshold_ms=15.0, min_dwell_time_s=0.0, ladder=DEMO_LADDER)
    )
    tracker = LatencyTracker(window_size=50, warmup_samples=0)

    sources = generate_synthetic_sources("/tmp/sg_sla_demo", count=4, n_frames=60, width=96, height=96)
    pipeline = MultiStreamPipeline(
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        postprocessor=Postprocessor(),
        tracker=tracker,
        model_input_size=640,
        queue_config=QueueConfig(),  # default: drop-oldest pre-inference, block post-inference
        batch_config=BatchConfig(max_batch_size=8, batch_timeout_ms=10.0),
        sla_controller=controller,
    )

    result = pipeline.run(sources, join_timeout=30)

    print(f"frames processed: {result['frames_processed']}")
    print(f"final tier: {controller.current_level} ({controller.current_tier_config.description})")
    print(f"backend actually loaded: {backend.current_config}")

    print("\ntier transitions:")
    for t in controller.transition_log:
        print(f"  {t.from_tier} -> {t.to_tier}  ({t.reason}, triggered by p99={t.triggering_latency_ms:.1f}ms)")

    tiers_seen = [s.tier for s in tracker.samples]
    print(f"\ntier per frame (first 20): {tiers_seen[:20]}")
    print(f"tier per frame (last 20):  {tiers_seen[-20:]}")


if __name__ == "__main__":
    main()
