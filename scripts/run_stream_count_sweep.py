#!/usr/bin/env python3
"""CLI: sweep stream count at a fixed degradation tier until the SLA
deadline is violated, writing one CSV row per stream count tried.

Runnable today against the mock backend (streamguard.inference.mock) to
verify the sweep logic itself; pass --backend tensorrt once stage 4 is
verified on real hardware for numbers that actually mean something.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os

from streamguard.config import DEFAULT_LADDER, DegradationTier, ModelTier, Precision
from streamguard.decode.cpu import CPUDecoder
from streamguard.experiments.sources import resolve_sources
from streamguard.experiments.stream_count_sweep import sweep_stream_counts
from streamguard.inference.mock import MockInferenceBackend
from streamguard.preprocess.cpu import CPUPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-level", type=int, default=0, help="ladder level to hold fixed (default: baseline)")
    parser.add_argument("--model-tier", default=None, help="override the ladder tier's model (e.g. tiny); needs --precision/--input-size")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--json-output", default=None, help="also write rows + device/driver/TRT/git metadata as JSON")
    parser.add_argument("--deadline-ms", type=float, default=33.0)
    parser.add_argument("--stream-counts", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32])
    parser.add_argument("--source-dir", default=None, help="real video files; default: generate synthetic clips")
    parser.add_argument("--work-dir", default="./sweep_work", help="scratch dir for synthetic clips (not committed)")
    parser.add_argument("--output", default="./results/stream_count_sweep.csv")
    parser.add_argument("--max-frames-per-stream", type=int, default=120)
    parser.add_argument("--backend", choices=["mock", "tensorrt"], default="mock")
    parser.add_argument("--engine-dir", default="./engines", help="required when --backend tensorrt")
    args = parser.parse_args()

    tier = next(t for t in DEFAULT_LADDER if t.level == args.tier_level)
    if args.model_tier:
        if not (args.precision and args.input_size):
            parser.error("--model-tier requires --precision and --input-size")
        tier = DegradationTier(
            level=tier.level, model_tier=ModelTier(args.model_tier), precision=Precision(args.precision),
            input_size=args.input_size, description="sweep override",
        )
    sources = resolve_sources(
        count=max(args.stream_counts), output_dir=args.work_dir, source_dir=args.source_dir,
    )

    if args.backend == "tensorrt":
        from streamguard.inference.tensorrt_backend import TensorRTInferenceBackend

        backend = TensorRTInferenceBackend(engine_dir=args.engine_dir)
    else:
        backend = MockInferenceBackend()

    results = sweep_stream_counts(
        stream_counts=args.stream_counts,
        tier=tier,
        sources=sources,
        decoder_factory=CPUDecoder,
        preprocessor=CPUPreprocessor(),
        inference_backend=backend,
        deadline_ms=args.deadline_ms,
        max_frames_per_stream=args.max_frames_per_stream,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(results[0])])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))

    from streamguard.experiments.stream_count_sweep import max_sustainable_stream_count

    max_n = max_sustainable_stream_count(results)
    if args.json_output:
        import json
        import subprocess
        import time

        from streamguard.metrics.device import capture_device_info

        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        payload = {
            "stage": "stream-count-sweep",
            "timestamp": time.time(),
            "git_commit": commit,
            "device": capture_device_info("cuda:0" if args.backend == "tensorrt" else "cpu").__dict__,
            "backend": args.backend,
            "tier": {"level": tier.level, "model": tier.model_tier.value, "precision": tier.precision.value,
                     "input_size": tier.input_size},
            "deadline_ms": args.deadline_ms,
            "decode": "cpu (cv2)",
            "max_sustainable_stream_count": max_n,
            "rows": [dataclasses.asdict(r) for r in results],
        }
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_output}")
    print(f"wrote {args.output} ({len(results)} rows)")
    print(f"max sustainable stream count at tier {tier.level} ({tier.model_tier.value}/{tier.precision.value}): {max_n}")


if __name__ == "__main__":
    main()
