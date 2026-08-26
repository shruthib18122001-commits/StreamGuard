import numpy as np
import pytest

from streamguard.metrics.device import capture_device_info
from streamguard.metrics.events import TierTransition, TransitionLog
from streamguard.metrics.latency import FrameLatency, LatencyTracker


def make_sample(total_ms: float, stream_id: int = 0, tier: int = 0) -> FrameLatency:
    return FrameLatency(
        decode_ms=total_ms * 0.1,
        preprocess_ms=total_ms * 0.1,
        inference_ms=total_ms * 0.6,
        postprocess_ms=total_ms * 0.2,
        total_ms=total_ms,
        stream_id=stream_id,
        tier=tier,
    )


def test_warmup_samples_are_discarded():
    tracker = LatencyTracker(window_size=100, warmup_samples=5)
    for i in range(5):
        tracker.record(make_sample(1000.0))  # would blow up percentiles if counted
    assert tracker.count == 0
    assert not tracker.is_warmed_up

    tracker.record(make_sample(10.0))
    assert tracker.count == 1
    assert tracker.is_warmed_up
    assert tracker.total_seen == 6


def test_percentiles_match_numpy_reference():
    tracker = LatencyTracker(window_size=1000, warmup_samples=0)
    values = list(range(1, 1001))  # 1..1000 ms
    for v in values:
        tracker.record(make_sample(float(v)))

    expected_p99 = float(np.percentile(values, 99))
    expected_p50 = float(np.percentile(values, 50))
    assert tracker.p99() == pytest.approx(expected_p99)
    assert tracker.p50() == pytest.approx(expected_p50)


def test_window_is_bounded_and_drops_oldest():
    tracker = LatencyTracker(window_size=10, warmup_samples=0)
    for v in range(1, 21):  # 1..20, window should only keep the last 10
        tracker.record(make_sample(float(v)))
    assert tracker.count == 10
    assert tracker.mean() == pytest.approx(np.mean(list(range(11, 21))))


def test_stage_breakdown_has_all_stages():
    tracker = LatencyTracker(window_size=50, warmup_samples=0)
    for v in range(1, 11):
        tracker.record(make_sample(float(v)))
    breakdown = tracker.stage_breakdown()
    assert set(breakdown.keys()) == {"decode", "preprocess", "inference", "postprocess", "total"}
    for stage_stats in breakdown.values():
        assert stage_stats["median"] is not None
        assert stage_stats["p99"] is not None


def test_empty_tracker_returns_none_not_error():
    tracker = LatencyTracker(window_size=10, warmup_samples=0)
    assert tracker.p50() is None
    assert tracker.mean() is None
    assert tracker.variance() is None


def test_invalid_stage_name_raises():
    tracker = LatencyTracker(window_size=10, warmup_samples=0)
    tracker.record(make_sample(5.0))
    with pytest.raises(ValueError):
        tracker.percentile(50, stage="not_a_stage")


def test_transition_log_records_in_order():
    log = TransitionLog()
    log.record(TierTransition(timestamp=1.0, from_tier=0, to_tier=1, triggering_latency_ms=40.0, reason="degrade"))
    log.record(TierTransition(timestamp=5.0, from_tier=1, to_tier=0, triggering_latency_ms=15.0, reason="recover"))
    assert len(log) == 2
    dicts = log.to_dicts()
    assert dicts[0]["reason"] == "degrade"
    assert dicts[1]["reason"] == "recover"


def test_device_info_capture_does_not_raise_without_gpu():
    info = capture_device_info("cuda:0")
    assert info.device == "cuda:0"
    # On a machine with no NVIDIA driver / no pynvml or tensorrt installed,
    # these must gracefully come back None rather than raising.
    assert info.captured_at > 0
