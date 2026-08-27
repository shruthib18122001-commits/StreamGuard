"""Exposes live pipeline state as Prometheus gauges: latency percentiles,
frames processed, current degradation tier, and per-queue depth/drop
counts. This is what the Grafana dashboard (docker/grafana/dashboard.json)
actually scrapes -- the live latency histogram, queue depth, and
degradation-tier panels CLAUDE.md's deliverables ask for.

Uses a private CollectorRegistry rather than the client library's global
default, so creating more than one PrometheusExporter in the same process
(e.g. across tests) never collides on duplicate metric names.
"""
from __future__ import annotations

from typing import Dict

from prometheus_client import CollectorRegistry, Gauge, generate_latest, start_http_server

from streamguard.bounded_queue import BoundedQueue
from streamguard.metrics.latency import LatencyTracker
from streamguard.sla_controller import SLAController


class PrometheusExporter:
    def __init__(self):
        self.registry = CollectorRegistry()
        self._p50 = Gauge("streamguard_latency_p50_ms", "p50 end-to-end latency (ms)", registry=self.registry)
        self._p95 = Gauge("streamguard_latency_p95_ms", "p95 end-to-end latency (ms)", registry=self.registry)
        self._p99 = Gauge("streamguard_latency_p99_ms", "p99 end-to-end latency (ms)", registry=self.registry)
        self._frames_processed = Gauge(
            "streamguard_frames_processed_total", "frames processed", registry=self.registry
        )
        self._degradation_tier = Gauge(
            "streamguard_degradation_tier", "current SLA controller tier level (0 = baseline)", registry=self.registry
        )
        self._queue_depth = Gauge(
            "streamguard_queue_depth", "current bounded queue depth", ["queue"], registry=self.registry
        )
        self._dropped_frames = Gauge(
            "streamguard_dropped_frames_total", "frames dropped by backpressure", ["queue"], registry=self.registry
        )

    def start_server(self, port: int = 9108) -> None:
        start_http_server(port, registry=self.registry)

    def update_from_tracker(self, tracker: LatencyTracker) -> None:
        for gauge, value in ((self._p50, tracker.p50()), (self._p95, tracker.p95()), (self._p99, tracker.p99())):
            if value is not None:
                gauge.set(value)
        self._frames_processed.set(tracker.total_seen)

    def update_tier(self, controller: SLAController) -> None:
        self._degradation_tier.set(controller.current_level)

    def update_queue(self, name: str, queue: BoundedQueue) -> None:
        self._queue_depth.labels(queue=name).set(len(queue))
        self._dropped_frames.labels(queue=name).set(queue.dropped_count)

    def render_text(self) -> str:
        """The exposition-format text a scrape would receive -- used by
        tests to check metric values without opening a real HTTP port."""
        return generate_latest(self.registry).decode("utf-8")
