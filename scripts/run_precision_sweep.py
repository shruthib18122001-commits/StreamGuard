#!/usr/bin/env python3
"""CLI: run every rung of the degradation ladder at a fixed stream count,
recording latency/throughput/power, writing one CSV row per tier.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os

from streamguard.config import DEFAULT_LADDER
from streamguard.decode.cpu import CPUDecoder
from streamguard.experiments.precision_sweep import sweep_precision
from streamguard.experiments.sources import resolve_sources
from streamguard.inference.mock import MockInferenceBackend
from streamguard.metrics.nvml_sampler import NVMLSampler
from streamguard.preprocess.cpu import CPUPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-count", type=int, default=4)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--work-dir", default="./sweep_work")
    parser.add_argument("--output", default="./results/precision_sweep.csv")
    parser.add_argument("--max-frames-per-stream", type=int, default=120)
    parser.add_argument("--sample-power", action="store_true", help="bracket each trial with NVMLSampler")
    parser.add_argument("--backend", choices=["mock", "tensorrt"], default="mock")
    parser.add_argument("--engine-dir", default="./engines", help="required when --backend tensorrt")
    args = parser.parse_args()

    sources = resolve_sources(count=args.stream_count, output_dir=args.work_dir, source_dir=args.source_dir)
    if args.backend == "tensorrt":
        from streamguard.inference.tensorrt_backend import TensorRTInferenceBackend

        backend = TensorRTInferenceBackend(engine_dir=args.engine_dir)
    else:
        backend = MockInferenceBackend()
    nvml = NVMLSampler() if args.sample_power else None

    results = sweep_precision(
        ladder=DEFAULT_LADDER,
        stream_count=args.stream_count,
        sources=sources,
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        max_frames_per_stream=args.max_frames_per_stream,
        nvml_sampler=nvml,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(results[0])])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))

    print(f"wrote {args.output} ({len(results)} rows)")
    if nvml is not None and nvml.unavailable_reason:
        print(f"NVML sampling unavailable: {nvml.unavailable_reason}")


if __name__ == "__main__":
    main()
