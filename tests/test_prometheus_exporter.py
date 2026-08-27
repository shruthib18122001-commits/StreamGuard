from streamguard.bounded_queue import BoundedQueue
from streamguard.config import BackpressurePolicy, ModelTier, Precision, SLAConfig
from streamguard.metrics.latency import FrameLatency, LatencyTracker
from streamguard.metrics.prometheus_exporter import PrometheusExporter
from streamguard.sla_controller import SLAController


def test_update_from_tracker_exposes_percentiles():
    exporter = PrometheusExporter()
    tracker = LatencyTracker(window_size=50, warmup_samples=0)
    for v in range(1, 21):
        tracker.record(FrameLatency(decode_ms=0, preprocess_ms=0, inference_ms=v, postprocess_ms=0, total_ms=float(v)))

    exporter.update_from_tracker(tracker)
    text = exporter.render_text()

    assert "streamguard_latency_p50_ms" in text
    assert "streamguard_latency_p99_ms" in text
    assert "streamguard_frames_processed_total 20.0" in text


def test_update_from_tracker_before_any_data_does_not_crash():
    exporter = PrometheusExporter()
    tracker = LatencyTracker(window_size=50, warmup_samples=0)

    exporter.update_from_tracker(tracker)  # no samples yet -- percentiles are None
    text = exporter.render_text()

    assert "streamguard_frames_processed_total 0.0" in text


def test_update_tier_reflects_controller_level():
    from streamguard.config import DegradationTier

    exporter = PrometheusExporter()
    controller = SLAController(
        SLAConfig(
            degrade_threshold_ms=1.0,
            recover_threshold_ms=0.5,
            min_dwell_time_s=0.0,
            ladder=[
                DegradationTier(level=0, model_tier=ModelTier.M, precision=Precision.FP16, input_size=640),
                DegradationTier(level=1, model_tier=ModelTier.NANO, precision=Precision.INT8, input_size=416),
            ],
        )
    )
    tracker = LatencyTracker(window_size=10, warmup_samples=0)
    for _ in range(5):
        tracker.record(FrameLatency(decode_ms=0, preprocess_ms=0, inference_ms=100, postprocess_ms=0, total_ms=100.0))
    controller.maybe_transition(tracker, now=1000.0)  # degrades 0 -> 1

    exporter.update_tier(controller)
    text = exporter.render_text()

    assert "streamguard_degradation_tier 1.0" in text


def test_update_queue_reports_depth_and_drops():
    exporter = PrometheusExporter()
    q = BoundedQueue(maxsize=2, policy=BackpressurePolicy.DROP_OLDEST)
    q.put(1)
    q.put(2)
    q.put(3)  # evicts 1, dropped_count -> 1

    exporter.update_queue("decode_out", q)
    text = exporter.render_text()

    assert 'streamguard_queue_depth{queue="decode_out"} 2.0' in text
    assert 'streamguard_dropped_frames_total{queue="decode_out"} 1.0' in text


def test_two_exporters_in_the_same_process_do_not_collide():
    # Each uses its own CollectorRegistry -- this must not raise a
    # duplicate-metric-name error, unlike registering twice against the
    # client library's global default registry.
    PrometheusExporter()
    PrometheusExporter()
