import numpy as np
import pytest

from streamguard.config import BatchConfig, DegradationTier, ModelTier, Precision, QueueConfig
from streamguard.decode.cpu import CPUDecoder
from streamguard.evaluation.map_eval import GroundTruthBox
from streamguard.experiments.decode_comparison import compare_decoders, measure_decode_throughput
from streamguard.experiments.map_trial import run_map_trial, sweep_map_per_tier
from streamguard.experiments.precision_sweep import run_precision_trial, sweep_precision
from streamguard.experiments.sources import generate_synthetic_sources, resolve_sources
from streamguard.experiments.stream_count_sweep import (
    max_sustainable_stream_count,
    run_stream_count_trial,
    sweep_stream_counts,
)
from streamguard.inference.mock import MockInferenceBackend
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor

SMALL_LADDER = [
    DegradationTier(level=0, model_tier=ModelTier.M, precision=Precision.FP16, input_size=640),
    DegradationTier(level=1, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416),
]


def test_generate_synthetic_sources_creates_requested_count(tmp_path):
    paths = generate_synthetic_sources(str(tmp_path), count=3, n_frames=5)
    assert len(paths) == 3
    for p in paths:
        assert p.endswith(".mp4")


def test_resolve_sources_without_source_dir_generates_synthetic(tmp_path):
    paths = resolve_sources(count=2, output_dir=str(tmp_path), n_frames=5)
    assert len(paths) == 2


def test_resolve_sources_reuses_real_files_round_robin(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    generate_synthetic_sources(str(video_dir), count=2, n_frames=5)

    resolved = resolve_sources(count=5, output_dir=str(tmp_path / "unused"), source_dir=str(video_dir))
    assert len(resolved) == 5
    assert len(set(resolved)) == 2  # only 2 real files, reused round-robin


def test_resolve_sources_raises_on_empty_source_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        resolve_sources(count=2, output_dir=str(tmp_path), source_dir=str(empty_dir))


def test_run_stream_count_trial_reports_frame_count_and_percentiles(tmp_path):
    from streamguard.config import BackpressurePolicy

    sources = generate_synthetic_sources(str(tmp_path), count=1, n_frames=8, width=64, height=64)
    backend = MockInferenceBackend(latency_table={(ModelTier.NANO, Precision.INT8): 1.0}, jitter_ms=0.0)
    tier = DegradationTier(level=0, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416)

    trial = run_stream_count_trial(
        stream_count=2,
        tier=tier,
        sources=sources,
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        deadline_ms=1000.0,
        warmup_samples=0,
        max_frames_per_stream=8,
        # Lossless queues: this test checks frame accounting, not
        # backpressure -- the default drop-oldest policy would legitimately
        # (and correctly) discard some frames here, same reasoning as
        # test_multi_stream_pipeline.py's lossless_queue_config().
        queue_config=QueueConfig(
            decode_queue_policy=BackpressurePolicy.BLOCK,
            batch_queue_policy=BackpressurePolicy.BLOCK,
            postprocess_queue_policy=BackpressurePolicy.BLOCK,
        ),
    )

    assert trial.stream_count == 2
    assert trial.frames_processed == 16
    assert trial.p50_ms is not None and trial.p99_ms is not None
    assert trial.sla_violated is False


def test_stream_count_sweep_stops_at_first_violation(tmp_path):
    sources = generate_synthetic_sources(str(tmp_path), count=2, n_frames=10, width=64, height=64)
    backend = MockInferenceBackend(latency_table={(ModelTier.NANO, Precision.INT8): 5.0}, jitter_ms=0.0)
    tier = DegradationTier(level=0, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416)

    results = sweep_stream_counts(
        stream_counts=[1, 2, 3, 4],
        tier=tier,
        sources=sources,
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        deadline_ms=0.001,  # absurdly tight -- guarantees a violation on the very first trial
        warmup_samples=0,
        max_frames_per_stream=5,
    )

    assert len(results) == 1  # stopped immediately at the first (and only) violation
    assert results[0].sla_violated is True
    assert max_sustainable_stream_count(results) is None


def test_stream_count_sweep_all_pass_when_deadline_is_generous(tmp_path):
    sources = generate_synthetic_sources(str(tmp_path), count=2, n_frames=10, width=64, height=64)
    backend = MockInferenceBackend(latency_table={(ModelTier.NANO, Precision.INT8): 1.0}, jitter_ms=0.0)
    tier = DegradationTier(level=0, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416)

    results = sweep_stream_counts(
        stream_counts=[1, 2],
        tier=tier,
        sources=sources,
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        deadline_ms=10_000.0,  # absurdly generous -- guarantees no violation
        warmup_samples=0,
        max_frames_per_stream=5,
    )

    assert len(results) == 2
    assert all(not r.sla_violated for r in results)
    assert max_sustainable_stream_count(results) == 2


def test_precision_sweep_runs_one_trial_per_ladder_rung(tmp_path):
    sources = generate_synthetic_sources(str(tmp_path), count=1, n_frames=5, width=64, height=64)
    backend = MockInferenceBackend(
        latency_table={(ModelTier.M, Precision.FP16): 2.0, (ModelTier.NANO, Precision.INT8): 1.0},
        jitter_ms=0.0,
    )

    results = sweep_precision(
        ladder=SMALL_LADDER,
        stream_count=1,
        sources=sources,
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        warmup_samples=0,
        max_frames_per_stream=5,
    )

    assert len(results) == 2
    assert results[0].tier_level == 0
    assert results[1].tier_level == 1
    assert all(r.frames_processed == 5 for r in results)
    assert all(r.throughput_fps is not None and r.throughput_fps > 0 for r in results)


def test_map_trial_runs_against_mock_backend():
    backend = MockInferenceBackend(rng_seed=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    labeled_frames = {"img1": frame}
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=10, y1=10, x2=50, y2=50)]}

    result = run_map_trial(
        tier=SMALL_LADDER[0],
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        postprocessor=Postprocessor(),
        labeled_frames=labeled_frames,
        ground_truths=ground_truths,
        num_classes=1,
    )

    assert "map" in result
    assert backend.current_config["tier"] == ModelTier.M


def test_sweep_map_per_tier_covers_every_rung():
    backend = MockInferenceBackend(rng_seed=3)
    labeled_frames = {"img1": np.zeros((100, 100, 3), dtype=np.uint8)}
    ground_truths = {"img1": [GroundTruthBox(class_id=0, x1=10, y1=10, x2=50, y2=50)]}

    results = sweep_map_per_tier(
        ladder=SMALL_LADDER,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        postprocessor=Postprocessor(),
        labeled_frames=labeled_frames,
        ground_truths=ground_truths,
        num_classes=1,
    )

    assert set(results.keys()) == {0, 1}


def test_decode_throughput_measures_real_cpu_decode(tmp_path):
    sources = generate_synthetic_sources(str(tmp_path), count=1, n_frames=20, width=64, height=64)
    result = measure_decode_throughput(CPUDecoder, sources[0], decoder_name="cpu")

    assert result.frames_decoded == 20
    assert result.fps > 0
    assert result.wall_seconds > 0


def test_compare_decoders_runs_each_named_decoder(tmp_path):
    sources = generate_synthetic_sources(str(tmp_path), count=1, n_frames=10, width=64, height=64)
    results = compare_decoders([("cpu-a", CPUDecoder), ("cpu-b", CPUDecoder)], sources[0])

    assert [r.decoder_name for r in results] == ["cpu-a", "cpu-b"]
    assert all(r.frames_decoded == 10 for r in results)
