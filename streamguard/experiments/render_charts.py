"""Renders the throughput/latency/power frontier charts CLAUDE.md asks
for, from the CSVs the sweep scripts write. Matplotlib, headless (Agg) --
no display needed, so this runs the same in a terminal, CI, or a Colab
cell.

Colors follow the dataviz skill's validated default palette
(references/palette.md): categorical slots 1-3 (blue/orange/aqua) for the
three percentile lines -- documented to clear every CVD/contrast gate as a
set of three, comfortable without a validator re-run. The degradation
ladder gets the single-hue ORDINAL ramp (light-to-dark blue), not
categorical colors: tiers are a genuinely ordered progression (quality
descending), and one hue ramping darker communicates that order the way
six unrelated categorical hues would not.

No dual-axis charts anywhere -- two measures of different scale become two
figures (or the frontier scatter, which plots two real, independently-
scaled axes against each other; that's a standard 2D plot, not the
disallowed two-y-scales-one-x-axis form).

matplotlib is an optional dependency (`pip install -e ".[viz]"`), imported
lazily inside each plotting function rather than at module level -- so
`import streamguard.experiments` (and every sweep script that has nothing
to do with charts) doesn't require it to be installed at all.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

# Chart chrome & ink -- light mode, from the dataviz skill's reference palette.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots 1-3: documented to clear every gate as a set of three.
COLOR_P50 = "#2a78d6"  # slot 1, blue
COLOR_P95 = "#eb6834"  # slot 2, orange
COLOR_P99 = "#1baf7a"  # slot 3, aqua

# Ordinal single-hue ramp (light -> dark blue) for the 6 ordered ladder tiers.
TIER_RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#104281"]

STATUS_CRITICAL = "#d03b3b"

_plt = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plt = plt
    return _plt


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _tier_label(row: Dict[str, Any]) -> str:
    return f"{row['tier_level']}: {row.get('model_tier', '')}/{row.get('precision', '')}"


def plot_latency_vs_stream_count(csv_path: str, deadline_ms: float, output_path: str) -> None:
    plt = _get_plt()
    rows = _read_csv(csv_path)
    n = [int(r["stream_count"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)
    for key, color, label in [("p50_ms", COLOR_P50, "p50"), ("p95_ms", COLOR_P95, "p95"), ("p99_ms", COLOR_P99, "p99")]:
        values = [float(r[key]) if r[key] else None for r in rows]
        ax.plot(n, values, color=color, linewidth=2, marker="o", markersize=6, label=label)
    ax.axhline(deadline_ms, color=STATUS_CRITICAL, linewidth=1.5, linestyle="--", label=f"deadline ({deadline_ms:.0f}ms)")

    ax.set_xlabel("concurrent streams", color=INK_SECONDARY)
    ax.set_ylabel("latency (ms)", color=INK_SECONDARY)
    ax.set_title("Latency vs. stream count", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_throughput_vs_tier(csv_path: str, output_path: str) -> None:
    plt = _get_plt()
    rows = _read_csv(csv_path)
    labels = [_tier_label(r) for r in rows]
    values = [float(r["throughput_fps"]) if r["throughput_fps"] else 0.0 for r in rows]
    colors = [TIER_RAMP[i % len(TIER_RAMP)] for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)
    ax.bar(labels, values, color=colors, width=0.6, zorder=2)
    ax.set_ylabel("throughput (fps)", color=INK_SECONDARY)
    ax.set_title("Throughput per degradation tier", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_latency_vs_tier(csv_path: str, deadline_ms: float, output_path: str) -> None:
    plt = _get_plt()
    rows = _read_csv(csv_path)
    labels = [_tier_label(r) for r in rows]
    values = [float(r["p99_ms"]) if r["p99_ms"] else 0.0 for r in rows]
    colors = [TIER_RAMP[i % len(TIER_RAMP)] for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)
    ax.bar(labels, values, color=colors, width=0.6, zorder=2)
    ax.axhline(deadline_ms, color=STATUS_CRITICAL, linewidth=1.5, linestyle="--", label=f"deadline ({deadline_ms:.0f}ms)")
    ax.set_ylabel("p99 latency (ms)", color=INK_SECONDARY)
    ax.set_title("p99 latency per degradation tier", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_power_vs_tier(csv_path: str, output_path: str) -> bool:
    """Returns False (and writes nothing) if no power samples exist --
    e.g. NVMLSampler was never available, which is the normal case on any
    machine without an NVIDIA driver."""
    plt = _get_plt()
    rows = _read_csv(csv_path)
    values = [float(r["mean_power_watts"]) if r.get("mean_power_watts") else None for r in rows]
    if all(v is None for v in values):
        return False

    labels = [_tier_label(r) for r in rows]
    colors = [TIER_RAMP[i % len(TIER_RAMP)] for i in range(len(rows))]
    plotted = [v if v is not None else 0.0 for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style_axes(ax)
    ax.bar(labels, plotted, color=colors, width=0.6, zorder=2)
    ax.set_ylabel("mean power (W)", color=INK_SECONDARY)
    ax.set_title("Power draw per degradation tier", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_frontier(csv_path: str, output_path: str) -> None:
    """Throughput vs. p99 latency, one point per tier, connected in tier
    order -- the actual frontier curve, not a dual-axis chart: both axes
    are independently scaled measurements of the same trials."""
    plt = _get_plt()
    rows = _read_csv(csv_path)
    x = [float(r["p99_ms"]) if r["p99_ms"] else 0.0 for r in rows]
    y = [float(r["throughput_fps"]) if r["throughput_fps"] else 0.0 for r in rows]
    colors = [TIER_RAMP[i % len(TIER_RAMP)] for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot(x, y, color=BASELINE, linewidth=1.5, zorder=2)
    ax.scatter(x, y, color=colors, s=90, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(str(r["tier_level"]), (xi, yi), textcoords="offset points", xytext=(6, 6), color=INK_SECONDARY, fontsize=9)

    ax.set_xlabel("p99 latency (ms)", color=INK_SECONDARY)
    ax.set_ylabel("throughput (fps)", color=INK_SECONDARY)
    ax.set_title("Throughput / latency frontier across the degradation ladder", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def render_all(results_dir: str, output_dir: str, deadline_ms: float = 33.0) -> List[str]:
    """Renders every chart whose input CSV exists in results_dir. Returns
    the list of output paths actually written."""
    os.makedirs(output_dir, exist_ok=True)
    written = []

    stream_csv = os.path.join(results_dir, "stream_count_sweep.csv")
    if os.path.exists(stream_csv):
        out = os.path.join(output_dir, "latency_vs_stream_count.png")
        plot_latency_vs_stream_count(stream_csv, deadline_ms, out)
        written.append(out)

    precision_csv = os.path.join(results_dir, "precision_sweep.csv")
    if os.path.exists(precision_csv):
        out = os.path.join(output_dir, "throughput_vs_tier.png")
        plot_throughput_vs_tier(precision_csv, out)
        written.append(out)

        out = os.path.join(output_dir, "latency_vs_tier.png")
        plot_latency_vs_tier(precision_csv, deadline_ms, out)
        written.append(out)

        out = os.path.join(output_dir, "power_vs_tier.png")
        if plot_power_vs_tier(precision_csv, out):
            written.append(out)

        out = os.path.join(output_dir, "frontier.png")
        plot_frontier(precision_csv, out)
        written.append(out)

    return written
