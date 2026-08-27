"""Structural checks on the Docker/Prometheus/Grafana deployment config.
Can't verify these actually run (no Docker on this dev machine -- see
README's Colab/deployment checklist), but a syntax or a metric-name typo
that would silently break the dashboard is exactly the kind of thing worth
catching here.
"""
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_docker_compose_defines_expected_services():
    with open(REPO_ROOT / "docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    assert set(compose["services"].keys()) == {"streamguard", "prometheus", "grafana"}
    assert compose["services"]["streamguard"]["ports"] == ["9108:9108"]


def test_prometheus_config_scrapes_streamguard_metrics_port():
    with open(REPO_ROOT / "docker/prometheus/prometheus.yml") as f:
        prom_config = yaml.safe_load(f)
    targets = prom_config["scrape_configs"][0]["static_configs"][0]["targets"]
    assert targets == ["streamguard:9108"]


def test_grafana_datasource_points_at_prometheus_service():
    with open(REPO_ROOT / "docker/grafana/provisioning/datasources/prometheus.yml") as f:
        ds_config = yaml.safe_load(f)
    assert ds_config["datasources"][0]["url"] == "http://prometheus:9090"


def test_dashboard_json_is_valid_and_covers_required_panels():
    with open(REPO_ROOT / "docker/grafana/dashboard.json") as f:
        dashboard = json.load(f)

    titles = [p["title"].lower() for p in dashboard["panels"]]
    # CLAUDE.md's deliverable: "live latency histogram, queue depth, degradation tier"
    assert any("latency" in t for t in titles)
    assert any("queue depth" in t for t in titles)
    assert any("degradation tier" in t for t in titles)


def test_dashboard_panel_queries_reference_real_exporter_metric_names():
    from streamguard.bounded_queue import BoundedQueue
    from streamguard.metrics.prometheus_exporter import PrometheusExporter

    exporter = PrometheusExporter()
    # Labeled metrics (queue depth, dropped frames) only materialize in the
    # exposition text once a label combination has actually been set --
    # exercise that path so this check sees every metric name the running
    # exporter would eventually emit, not just the always-present gauges.
    exporter.update_queue("decode_out", BoundedQueue(maxsize=1))
    exported_text = exporter.render_text()
    exported_metric_names = {
        line.split("{")[0].split(" ")[0]
        for line in exported_text.splitlines()
        if line and not line.startswith("#")
    }

    with open(REPO_ROOT / "docker/grafana/dashboard.json") as f:
        dashboard = json.load(f)

    referenced = set()
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            referenced.add(target["expr"])

    # every PromQL expression in the dashboard must be a metric this
    # exporter actually emits -- a renamed gauge here would otherwise
    # silently break a dashboard panel with no test ever catching it
    assert referenced <= exported_metric_names, referenced - exported_metric_names
