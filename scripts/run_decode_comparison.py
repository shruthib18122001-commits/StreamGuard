#!/usr/bin/env python3
"""CLI: compare decode-engine throughput and CPU utilization. Only
CPUDecoder is runnable here; add NVDECDecoder to --decoders once it's
verified on Colab (README's GPU-only checklist covers that).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os

from streamguard.decode.cpu import CPUDecoder
from streamguard.experiments.decode_comparison import compare_decoders
from streamguard.experiments.sources import resolve_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--work-dir", default="./sweep_work")
    parser.add_argument("--output", default="./results/decode_comparison.csv")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    sources = resolve_sources(count=1, output_dir=args.work_dir, source_dir=args.source_dir, n_frames=args.max_frames)

    decoders = [("cpu", CPUDecoder)]
    try:
        from streamguard.decode.nvdec import NVDECDecoder

        import PyNvCodec  # noqa: F401 -- only add NVDEC to the comparison if it's actually importable here

        decoders.append(("nvdec", NVDECDecoder))
    except ImportError:
        print("PyNvCodec not available -- comparing CPU decode only (expected on this dev machine)")

    results = compare_decoders(decoders, sources[0], max_frames=args.max_frames)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(results[0])])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))

    print(f"wrote {args.output} ({len(results)} rows)")
    for r in results:
        print(f"  {r.decoder_name}: {r.fps:.1f} fps, {r.mean_cpu_percent}% CPU")


if __name__ == "__main__":
    main()
