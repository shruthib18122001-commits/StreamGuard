import pytest

from streamguard.config import DegradationTier, ModelTier, Precision, SLAConfig
from streamguard.decode.cpu import CPUDecoder
from streamguard.inference.mock import MockInferenceBackend
from streamguard.metrics.latency import LatencyTracker
from streamguard.pipeline import SingleStreamPipeline
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor
from streamguard.sla_controller import SLAController


def build_pipeline(tracker: LatencyTracker, stream_id: int = 0, rng_seed: int = 7) -> SingleStreamPipeline:
    backend = MockInferenceBackend(rng_seed=rng_seed)
    backend.load(ModelTier.NANO, Precision.INT8, 416)
    return SingleStreamPipeline(
        decoder=CPUDecoder(),
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        postprocessor=Postprocessor(),
        tracker=tracker,
        model_input_size=416,
        stream_id=stream_id,
    )


def test_pipeline_processes_every_frame(synthetic_video):
    path, n_frames, _, _ = synthetic_video
    tracker = LatencyTracker(window_size=100, warmup_samples=0)

    processed = build_pipeline(tracker).run(path)

    assert processed == n_frames
    assert tracker.count == n_frames


def test_pipeline_records_positive_per_stage_latency(synthetic_video):
    path, _, _, _ = synthetic_video
    tracker = LatencyTracker(window_size=100, warmup_samples=0)
    build_pipeline(tracker).run(path)

    for sample in tracker.samples:
        assert sample.decode_ms >= 0
        assert sample.preprocess_ms >= 0
        assert sample.inference_ms > 0  # mock backend always sleeps a non-zero amount
        assert sample.postprocess_ms >= 0
        assert sample.total_ms == pytest.approx(
            sample.decode_ms + sample.preprocess_ms + sample.inference_ms + sample.postprocess_ms
        )


def test_pipeline_respects_max_frames(synthetic_video):
    path, _, _, _ = synthetic_video
    tracker = LatencyTracker(window_size=100, warmup_samples=0)

    processed = build_pipeline(tracker).run(path, max_frames=5)

    assert processed == 5
    assert tracker.count == 5


def test_pipeline_tags_every_sample_with_stream_id(synthetic_video):
    path, _, _, _ = synthetic_video
    tracker = LatencyTracker(window_size=100, warmup_samples=0)

    build_pipeline(tracker, stream_id=3).run(path, max_frames=3)

    assert all(sample.stream_id == 3 for sample in tracker.samples)


def test_pipeline_tier_stays_zero_without_sla_controller(synthetic_video):
    path, _, _, _ = synthetic_video
    tracker = LatencyTracker(window_size=100, warmup_samples=0)

    build_pipeline(tracker).run(path, max_frames=3)

    assert all(sample.tier == 0 for sample in tracker.samples)


def test_pipeline_with_sla_controller_degrades_under_sustained_overload(synthetic_video):
    # A backend where the baseline tier is deliberately far over deadline
    # and the fallback tier is comfortably under it -- with zero dwell time,
    # the controller should degrade on the very first sample and stay there.
    path, n_frames, _, _ = synthetic_video
    tracker = LatencyTracker(window_size=100, warmup_samples=0)
    backend = MockInferenceBackend(
        latency_table={
            (ModelTier.M, Precision.FP16): 100.0,
            (ModelTier.NANO, Precision.INT8): 1.0,
        },
        rng_seed=1,
    )
    backend.load(ModelTier.M, Precision.FP16, 640)
    controller = SLAController(
        SLAConfig(
            degrade_threshold_ms=33.0,
            recover_threshold_ms=10.0,
            min_dwell_time_s=0.0,
            ladder=[
                DegradationTier(level=0, model_tier=ModelTier.M, precision=Precision.FP16, input_size=640),
                DegradationTier(level=1, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416),
            ],
        )
    )
    pipeline = SingleStreamPipeline(
        decoder=CPUDecoder(),
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        postprocessor=Postprocessor(),
        tracker=tracker,
        model_input_size=640,
        sla_controller=controller,
    )

    pipeline.run(path)

    assert tracker.count == n_frames
    tiers_seen = [s.tier for s in tracker.samples]
    assert tiers_seen[0] == 0  # first frame still ran at baseline
    assert tiers_seen[-1] == 1  # degraded by the end
    assert len(controller.transition_log) >= 1
    assert controller.transition_log.entries[0].reason == "degrade"
