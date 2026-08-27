import cv2
import numpy as np
import pytest

from streamguard.config import BackpressurePolicy, BatchConfig, ModelTier, Precision, QueueConfig
from streamguard.decode.cpu import CPUDecoder
from streamguard.inference.mock import MockInferenceBackend
from streamguard.metrics.latency import LatencyTracker
from streamguard.multi_stream_pipeline import MultiStreamPipeline
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor


def make_video(tmp_path, name, n_frames, seed, width=160, height=120, fps=10):
    path = str(tmp_path / name)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rng = np.random.default_rng(seed)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8))
    writer.release()
    return path


def build_pipeline(queue_config=None, batch_config=None, backend=None):
    tracker = LatencyTracker(window_size=500, warmup_samples=0)
    backend = backend or MockInferenceBackend(rng_seed=11, jitter_ms=0.0)
    backend.load(ModelTier.NANO, Precision.INT8, 416)
    pipeline = MultiStreamPipeline(
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        postprocessor=Postprocessor(),
        tracker=tracker,
        model_input_size=416,
        queue_config=queue_config or QueueConfig(),
        batch_config=batch_config or BatchConfig(max_batch_size=8, batch_timeout_ms=50.0),
    )
    return pipeline, tracker


def lossless_queue_config() -> QueueConfig:
    # BLOCK everywhere: these tests check orchestration correctness (every
    # frame accounted for, batches spanning streams), not backpressure — the
    # default DROP_OLDEST policy would legitimately (and correctly) discard
    # frames here, since fast synthetic-video decode easily outpaces the
    # mock inference stage. That behavior has its own dedicated tests below.
    return QueueConfig(
        decode_queue_policy=BackpressurePolicy.BLOCK,
        batch_queue_policy=BackpressurePolicy.BLOCK,
        postprocess_queue_policy=BackpressurePolicy.BLOCK,
    )


def test_all_streams_fully_processed(tmp_path):
    sources = [
        make_video(tmp_path, "a.mp4", n_frames=12, seed=1),
        make_video(tmp_path, "b.mp4", n_frames=8, seed=2),
        make_video(tmp_path, "c.mp4", n_frames=15, seed=3),
    ]
    pipeline, tracker = build_pipeline(queue_config=lossless_queue_config())

    result = pipeline.run(sources, join_timeout=10)

    assert result["frames_processed"] == 12 + 8 + 15
    counts = {}
    for sample in tracker.samples:
        counts[sample.stream_id] = counts.get(sample.stream_id, 0) + 1
    assert counts == {0: 12, 1: 8, 2: 15}


def test_batching_combines_frames_across_streams(tmp_path):
    # Slow-ish mock inference + a generous batch timeout gives concurrent
    # decode threads from 4 streams time to land in the same batch.
    sources = [make_video(tmp_path, f"s{i}.mp4", n_frames=10, seed=i) for i in range(4)]
    backend = MockInferenceBackend(rng_seed=5, jitter_ms=0.0)
    pipeline, tracker = build_pipeline(
        queue_config=lossless_queue_config(),
        batch_config=BatchConfig(max_batch_size=8, batch_timeout_ms=50.0),
        backend=backend,
    )

    result = pipeline.run(sources, join_timeout=10)

    assert result["frames_processed"] == 40
    assert max(result["batch_sizes_seen"]) > 1, "expected at least one batch with frames from multiple streams"
    assert sum(result["batch_sizes_seen"]) == 40


def test_max_frames_per_stream_caps_each_stream(tmp_path):
    sources = [
        make_video(tmp_path, "a.mp4", n_frames=20, seed=1),
        make_video(tmp_path, "b.mp4", n_frames=20, seed=2),
    ]
    pipeline, tracker = build_pipeline()

    result = pipeline.run(sources, max_frames_per_stream=5, join_timeout=10)

    assert result["frames_processed"] == 10
    counts = {}
    for sample in tracker.samples:
        counts[sample.stream_id] = counts.get(sample.stream_id, 0) + 1
    assert counts == {0: 5, 1: 5}


def test_drop_oldest_policy_drops_under_sustained_backpressure(tmp_path):
    # A single fast-decoding stream well ahead of a deliberately slow
    # inference backend: with a tiny drop-oldest queue, frames must pile up
    # faster than the batcher can drain them, so some get evicted.
    sources = [make_video(tmp_path, "fast.mp4", n_frames=60, seed=9)]
    slow_backend = MockInferenceBackend(
        latency_table={(ModelTier.NANO, Precision.INT8): 25.0},
        jitter_ms=0.0,
    )
    pipeline, tracker = build_pipeline(
        queue_config=QueueConfig(
            decode_queue_size=2,
            batch_queue_size=2,
            postprocess_queue_size=32,
            decode_queue_policy=BackpressurePolicy.DROP_OLDEST,
            batch_queue_policy=BackpressurePolicy.DROP_OLDEST,
            postprocess_queue_policy=BackpressurePolicy.BLOCK,
        ),
        batch_config=BatchConfig(max_batch_size=4, batch_timeout_ms=5.0),
        backend=slow_backend,
    )

    result = pipeline.run(sources, join_timeout=15)

    assert result["dropped_decode_queue"] + result["dropped_batch_queue"] > 0
    assert result["frames_processed"] < 60  # some frames were skipped, not eventually processed


def test_block_policy_processes_every_frame_even_when_slow(tmp_path):
    # Same slow-backend setup, but BLOCK policy: decode paces to match the
    # bottleneck instead of dropping, so every frame is still processed —
    # it just takes longer, which is exactly the tradeoff being tested.
    sources = [make_video(tmp_path, "fast.mp4", n_frames=15, seed=9)]
    slow_backend = MockInferenceBackend(
        latency_table={(ModelTier.NANO, Precision.INT8): 10.0},
        jitter_ms=0.0,
    )
    pipeline, tracker = build_pipeline(
        queue_config=QueueConfig(
            decode_queue_size=1,
            batch_queue_size=1,
            postprocess_queue_size=32,
            decode_queue_policy=BackpressurePolicy.BLOCK,
            batch_queue_policy=BackpressurePolicy.BLOCK,
            postprocess_queue_policy=BackpressurePolicy.BLOCK,
        ),
        batch_config=BatchConfig(max_batch_size=1, batch_timeout_ms=5.0),
        backend=slow_backend,
    )

    result = pipeline.run(sources, join_timeout=15)

    assert result["frames_processed"] == 15
    assert result["dropped_decode_queue"] == 0
    assert result["dropped_batch_queue"] == 0
