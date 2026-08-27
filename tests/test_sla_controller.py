import pytest

from streamguard.config import DegradationTier, ModelTier, Precision, SLAConfig
from streamguard.metrics.latency import FrameLatency, LatencyTracker
from streamguard.sla_controller import SLAController

SMALL_LADDER = [
    DegradationTier(level=0, model_tier=ModelTier.M, precision=Precision.FP16, input_size=640),
    DegradationTier(level=1, model_tier=ModelTier.S, precision=Precision.FP16, input_size=640),
    DegradationTier(level=2, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416),
]


def make_controller(degrade=30.0, recover=20.0, dwell=2.0) -> SLAController:
    cfg = SLAConfig(
        degrade_threshold_ms=degrade,
        recover_threshold_ms=recover,
        min_dwell_time_s=dwell,
        ladder=SMALL_LADDER,
    )
    return SLAController(cfg)


def tracker_with_constant_latency(value_ms: float) -> LatencyTracker:
    tracker = LatencyTracker(window_size=50, warmup_samples=0)
    for _ in range(20):
        tracker.record(FrameLatency(decode_ms=0, preprocess_ms=0, inference_ms=value_ms, postprocess_ms=0, total_ms=value_ms))
    return tracker


def test_no_transition_before_tracker_has_data():
    controller = make_controller()
    empty_tracker = LatencyTracker(window_size=50, warmup_samples=0)
    assert controller.maybe_transition(empty_tracker, now=1000.0) is None
    assert controller.current_level == 0


def test_degrades_one_level_when_over_threshold():
    controller = make_controller()
    tracker = tracker_with_constant_latency(40.0)
    transition = controller.maybe_transition(tracker, now=1000.0)
    assert transition is not None
    assert transition.from_tier == 0
    assert transition.to_tier == 1
    assert transition.reason == "degrade"
    assert controller.current_level == 1
    assert controller.current_tier_config.model_tier == ModelTier.S


def test_does_not_degrade_past_max_level():
    controller = make_controller(dwell=0.0)
    tracker = tracker_with_constant_latency(40.0)
    controller.maybe_transition(tracker, now=1000.0)  # 0 -> 1
    controller.maybe_transition(tracker, now=1001.0)  # 1 -> 2
    assert controller.current_level == 2
    transition = controller.maybe_transition(tracker, now=1002.0)  # already at max
    assert transition is None
    assert controller.current_level == 2


def test_recovers_one_level_when_under_recovery_threshold():
    controller = make_controller(dwell=0.0)
    tracker_hot = tracker_with_constant_latency(40.0)
    controller.maybe_transition(tracker_hot, now=1000.0)  # 0 -> 1
    assert controller.current_level == 1

    tracker_cool = tracker_with_constant_latency(5.0)
    transition = controller.maybe_transition(tracker_cool, now=1001.0)
    assert transition is not None
    assert transition.reason == "recover"
    assert transition.from_tier == 1
    assert transition.to_tier == 0
    assert controller.current_level == 0


def test_does_not_recover_past_level_zero():
    controller = make_controller()
    tracker_cool = tracker_with_constant_latency(5.0)
    transition = controller.maybe_transition(tracker_cool, now=1000.0)
    assert transition is None
    assert controller.current_level == 0


def test_hysteresis_band_prevents_oscillation():
    # Between recover (20) and degrade (30) thresholds: neither condition
    # fires, which is the whole point of a separate band instead of one
    # threshold.
    controller = make_controller(degrade=30.0, recover=20.0, dwell=0.0)
    tracker_mid = tracker_with_constant_latency(25.0)
    assert controller.maybe_transition(tracker_mid, now=1000.0) is None
    assert controller.current_level == 0


def test_min_dwell_time_blocks_rapid_transitions():
    controller = make_controller(dwell=10.0)
    tracker_hot = tracker_with_constant_latency(40.0)
    first = controller.maybe_transition(tracker_hot, now=1000.0)
    assert first is not None
    assert controller.current_level == 1

    # Still well over threshold, but within the dwell window -- must hold.
    second = controller.maybe_transition(tracker_hot, now=1005.0)
    assert second is None
    assert controller.current_level == 1

    # Dwell time has now elapsed.
    third = controller.maybe_transition(tracker_hot, now=1011.0)
    assert third is not None
    assert controller.current_level == 2


def test_transition_log_accumulates_entries():
    controller = make_controller(dwell=0.0)
    tracker_hot = tracker_with_constant_latency(40.0)
    controller.maybe_transition(tracker_hot, now=1000.0)
    controller.maybe_transition(tracker_hot, now=1001.0)
    assert len(controller.transition_log) == 2
    reasons = [t.reason for t in controller.transition_log]
    assert reasons == ["degrade", "degrade"]
