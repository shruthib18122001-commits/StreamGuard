#!/usr/bin/env python3
"""Long-running service entrypoint: runs the multi-stream pipeline against
configured sources with the SLA controller active, exposing live
Prometheus metrics. This is what the Docker image's default command runs.

With no real streams.sources configured, loops over short synthetic clips
indefinitely rather than generating one giant fake "continuous" file --
good enough to watch the dashboard move, not a substitute for real footage.
"""
from __future__ import annotations

import argparse
import threading

from streamguard.config import load_config
from streamguard.decode.cpu import CPUDecoder
from streamguard.experiments.sources import resolve_sources
from streamguard.inference.mock import MockInferenceBackend
from streamguard.metrics.latency import LatencyTracker
from streamguard.metrics.prometheus_exporter import PrometheusExporter
from streamguard.multi_stream_pipeline import MultiStreamPipeline
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor
from streamguard.sla_controller import SLAController


def _export_loop(
    exporter: PrometheusExporter,
    tracker: LatencyTracker,
    controller: SLAController,
    stop_event: threading.Event,
    interval_s: float = 2.0,
) -> None:
    while not stop_event.wait(interval_s):
        exporter.update_from_tracker(tracker)
        exporter.update_tier(controller)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--metrics-port", type=int, default=9108)
    parser.add_argument("--work-dir", default="/tmp/streamguard_synthetic")
    parser.add_argument("--max-loops", type=int, default=None, help="stop after N loops over the sources (testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sources = resolve_sources(
        count=cfg.streams.count, output_dir=args.work_dir, source_dir=None, n_frames=300,
    ) if not cfg.streams.sources else cfg.streams.sources
    if not cfg.streams.sources:
        print(f"no streams.sources configured -- looping synthetic clips from {args.work_dir}")

    backend = MockInferenceBackend()  # swap for TensorRTInferenceBackend once stage 4 is verified on Colab
    controller = SLAController(cfg.sla)
    tracker = LatencyTracker(window_size=cfg.sla.window_size, warmup_samples=cfg.sla.warmup_samples)
    exporter = PrometheusExporter()
    exporter.start_server(args.metrics_port)
    print(f"metrics on :{args.metrics_port}/metrics")

    stop_event = threading.Event()
    export_thread = threading.Thread(
        target=_export_loop, args=(exporter, tracker, controller, stop_event), daemon=True
    )
    export_thread.start()

    pipeline = MultiStreamPipeline(
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        postprocessor=Postprocessor(),
        tracker=tracker,
        model_input_size=cfg.model.input_size,
        queue_config=cfg.queues,
        batch_config=cfg.batching,
        sla_controller=controller,
    )

    loops = 0
    try:
        while args.max_loops is None or loops < args.max_loops:
            result = pipeline.run(sources, join_timeout=60)
            loops += 1
            print(f"loop {loops}: {result}")
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
