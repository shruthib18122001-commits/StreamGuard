import csv
import importlib
import os

from streamguard.experiments.render_charts import render_all


def test_matplotlib_import_is_deferred_not_module_level():
    # matplotlib is an optional (`viz`) dependency -- streamguard.experiments
    # must stay importable without it. Reloading resets the module-level
    # cache to confirm nothing eagerly imported/cached matplotlib just by
    # importing this module (only calling a plot function should).
    import streamguard.experiments.render_charts as rc

    importlib.reload(rc)
    assert rc._plt is None


def write_stream_count_csv(path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stream_count", "tier_level", "p50_ms", "p95_ms", "p99_ms",
                "frames_processed", "dropped_decode_queue", "dropped_batch_queue", "sla_violated",
            ],
        )
        writer.writeheader()
        for n, p99 in [(1, 10.0), (2, 15.0), (4, 25.0)]:
            writer.writerow({
                "stream_count": n, "tier_level": 0, "p50_ms": p99 * 0.5, "p95_ms": p99 * 0.9, "p99_ms": p99,
                "frames_processed": 100, "dropped_decode_queue": 0, "dropped_batch_queue": 0, "sla_violated": False,
            })


def write_precision_csv(path, with_power: bool):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tier_level", "model_tier", "precision", "input_size",
                "p50_ms", "p95_ms", "p99_ms", "throughput_fps", "frames_processed", "mean_power_watts",
            ],
        )
        writer.writeheader()
        for level, fps, p99 in [(0, 100.0, 40.0), (1, 300.0, 20.0), (2, 600.0, 10.0)]:
            writer.writerow({
                "tier_level": level, "model_tier": "m", "precision": "fp16", "input_size": 640,
                "p50_ms": p99 * 0.5, "p95_ms": p99 * 0.9, "p99_ms": p99,
                "throughput_fps": fps, "frames_processed": 100,
                "mean_power_watts": (50.0 + level * 5) if with_power else "",
            })


def test_render_all_writes_expected_charts_with_power(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_stream_count_csv(results_dir / "stream_count_sweep.csv")
    write_precision_csv(results_dir / "precision_sweep.csv", with_power=True)

    written = render_all(str(results_dir), str(tmp_path / "charts"))

    names = {os.path.basename(p) for p in written}
    assert names == {
        "latency_vs_stream_count.png",
        "throughput_vs_tier.png",
        "latency_vs_tier.png",
        "power_vs_tier.png",
        "frontier.png",
    }
    for path in written:
        assert os.path.getsize(path) > 0


def test_render_all_skips_power_chart_without_data(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_precision_csv(results_dir / "precision_sweep.csv", with_power=False)

    written = render_all(str(results_dir), str(tmp_path / "charts"))

    names = {os.path.basename(p) for p in written}
    assert "power_vs_tier.png" not in names


def test_render_all_returns_empty_list_with_no_csvs(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    written = render_all(str(results_dir), str(tmp_path / "charts"))
    assert written == []
