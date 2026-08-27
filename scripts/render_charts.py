#!/usr/bin/env python3
"""CLI: render the throughput/latency/power frontier charts from whatever
sweep CSVs exist in --results-dir.
"""
from __future__ import annotations

import argparse

from streamguard.experiments.render_charts import render_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./results/charts")
    parser.add_argument("--deadline-ms", type=float, default=33.0)
    args = parser.parse_args()

    written = render_all(args.results_dir, args.output_dir, args.deadline_ms)
    if not written:
        print(f"no sweep CSVs found in {args.results_dir!r} -- run the sweep scripts first")
        return
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
