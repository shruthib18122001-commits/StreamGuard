import pytest

from streamguard.config import ModelTier, Precision
from streamguard.decode.cpu import CPUDecoder
from streamguard.inference.mock import MockInferenceBackend
from streamguard.metrics.latency import LatencyTracker
from streamguard.pipeline import SingleStreamPipeline
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor


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


def test_pipeline_tier_is_baseline_until_sla_controller_exists(synthetic_video):
    path, _, _, _ = synthetic_video
    tracker = LatencyTracker(window_size=100, warmup_samples=0)

    build_pipeline(tracker).run(path, max_frames=3)

    assert all(sample.tier == 0 for sample in tracker.samples)
