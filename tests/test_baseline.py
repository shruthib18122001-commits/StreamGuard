import numpy as np

from streamguard.config import ModelTier, Precision
from streamguard.decode.cpu import CPUDecoder
from streamguard.experiments.baseline import expected_bytes_per_frame, measure_stage_breakdown, summarize
from streamguard.experiments.sources import generate_synthetic_sources
from streamguard.inference.mock import MockInferenceBackend
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor


def test_expected_bytes_match_claude_md_numbers_at_640():
    b = expected_bytes_per_frame(640)
    assert b["num_anchors"] == 8400
    assert b["d2h_raw_output_float32"] == 8400 * 85 * 4  # 2,856,000 B ~= 2.9 MB
    assert b["h2d_float32_nchw"] == 4 * b["host_uint8_letterboxed"]


def test_expected_bytes_at_416():
    assert expected_bytes_per_frame(416)["num_anchors"] == 52 * 52 + 26 * 26 + 13 * 13


def test_summarize_reports_spread_not_one_number():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median"] == 3.0 and s["p99"] > s["p95"] > s["median"] and s["variance"] > 0 and s["n"] == 5


def test_measure_stage_breakdown_with_mock_backend(tmp_path):
    src = generate_synthetic_sources(str(tmp_path), count=1, n_frames=6)[0]
    backend = MockInferenceBackend()
    backend.load(ModelTier.NANO, Precision.FP16, 416)
    r = measure_stage_breakdown(
        CPUDecoder, src, CPUPreprocessor(), backend, Postprocessor(),
        input_size=416, batch_size=4, iterations=5, warmup=2,
    )
    assert r["iterations"] == 5 and r["warmup_discarded"] == 2  # 4*7 frames > 6: feeder must loop the clip
    total = r["per_batch_ms"]["total_ms"]["median"]
    parts = sum(r["per_batch_ms"][k]["median"] for k in ("decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms"))
    assert abs(total - parts) < 0.5 * total
    assert abs(r["per_frame_ms"]["total_ms"]["median"] - total / 4) < 1e-6
    assert "measured_bytes_per_frame" not in r  # mock moves no bytes; must not invent them


def test_single_stream_pipeline_uses_injected_inference_timer(tmp_path):
    from streamguard.metrics.latency import LatencyTracker
    from streamguard.metrics.timer import Timer
    from streamguard.pipeline import SingleStreamPipeline

    calls = []

    class FakeTimer(Timer):
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *a):
            return False

        @property
        def elapsed_ms(self):
            return 7.0

    src = generate_synthetic_sources(str(tmp_path), count=1, n_frames=4)[0]
    backend = MockInferenceBackend()
    backend.load(ModelTier.NANO, Precision.FP16, 416)
    tracker = LatencyTracker(window_size=10, warmup_samples=0)
    SingleStreamPipeline(
        CPUDecoder(), CPUPreprocessor(), backend, Postprocessor(), tracker, 416, inference_timer_factory=FakeTimer,
    ).run(src)
    assert len(calls) == 4
    assert tracker.p50("inference") == 7.0
