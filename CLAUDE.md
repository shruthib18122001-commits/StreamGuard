# StreamGuard — Real-Time Multi-Stream Inference with SLA Enforcement

## What this is
A GPU video pipeline that ingests N concurrent streams, runs object detection on
every frame, and holds a hard end-to-end latency deadline (33ms @ 30fps). Under
saturation it degrades quality in a controlled way rather than falling behind.
The adaptive degradation controller is the core of the project — detection itself
is commodity.

Portfolio project for NVIDIA internship applications. Measurement rigor and clean
systems design matter as much as the code running.

## Who I am
MS CS student. Strong Python, FastAPI, Docker, Kubernetes, PostgreSQL, distributed
systems. **New to CUDA, TensorRT, and DeepStream** — explain GPU-specific decisions
as you go; I need to defend every design choice in an interview.

## Hardware reality
- No local NVIDIA GPU. Develop on Mac, run on **Google Colab (T4)**.
- **Jetson Orin Nano 2** may be added later. Keep the code Jetson-clean: no x86-only
  assumptions, no hardcoded TRT plugin paths, no baked-in CUDA arch flags, device
  selection via config.
- Every result row must carry device, driver version, and TensorRT version.

## Locked design decisions
Do not revisit these without asking me.

- **Detector:** YOLOX (Apache 2.0 — no AGPL). Tiers: nano / tiny / s / m.
  Load behind an interface so a different detector is a config change.
- **Runtime target:** Colab T4 first, Jetson-clean. Do not write an untested
  aarch64/JetPack path yet.
- **Concurrency:** threads + CUDA streams. One thread per pipeline stage.
  - NOT process-per-stream — separate CUDA contexts kill cross-stream batching.
  - NOT asyncio — CUDA work isn't awaitable; it degenerates into a thread pool anyway.
  - GIL is released during NVDEC, cudaMemcpyAsync, and TRT execute. Therefore:
    **no heavy per-frame work in pure Python.** Postprocessing stays on GPU or
    vectorized NumPy.
- **Repo:** GitHub, committed per stage. Never commit model weights, .engine files,
  video, or result CSVs.

## Architecture

Source -> NVDEC decode -> GPU preprocess -> batch -> TensorRT -> postprocess -> out
                                                        ^
                                          SLA Controller (reads p99 latency)

- Bounded queues between every stage. Backpressure policy must be explicit and
  documented — unbounded queues turn overload into OOM.
- Decode on GPU via NVDEC; also implement a CPU decode path purely to measure the gap.
- Preprocess on GPU (CuPy/CUDA). Frame stays device-resident from decode to output.
- Dynamic batching aggregates frames across streams. Batch on size N **or** a
  timeout, whichever fires first. Document the throughput/latency tradeoff.

## The SLA controller — build this carefully
- Rolling window of end-to-end per-frame latency. Track **p99**, not mean.
- Degradation ladder, cheapest sacrifice first:
  0: YOLOX-m FP16 640px (baseline)
  1: -> YOLOX-s
  2: -> INT8
  3: -> YOLOX-tiny, 416px
  4: -> YOLOX-nano
  5: -> selective frame skipping
- **Hysteresis required.** Separate thresholds for degrade vs. recover, plus a
  minimum dwell time per tier. A single threshold oscillates.
- Log every tier transition with the latency that triggered it. This time series
  is a key output.
- Controller logic must be testable with a mocked inference backend — no GPU.

## Measurement rules — non-negotiable
- **CUDA events for GPU timing. Never time.time().** Host timing measures submission,
  not completion, because GPU work is async. Comment this in the code.
- **Pinned (page-locked) host memory** for H2D transfers, or cudaMemcpyAsync silently
  runs synchronously and overlap disappears.
- **One TensorRT execution context per worker thread.** Contexts are not thread-safe;
  the engine can be shared.
- Discard warmup iterations. Report median, p95, p99, and variance — never a single
  number.
- Per-stage latency breakdown: decode / preprocess / inference / postprocess / total.
- NVML sampled continuously: power, SM+memory clocks, temperature, throttle reasons,
  memory usage. Needed to distinguish "code got slower" from "GPU throttled".
- mAP measured at every degradation tier against a labeled sample set.

## Experiments to automate
- Sweep stream count upward until SLA violation; find max sustainable N per tier.
- NVDEC vs CPU decode: throughput and CPU utilization.
- Precision sweep: FP32/FP16/INT8 on latency, throughput, power, mAP.
- Output the throughput / latency / accuracy / power frontier as charts.

## Deliverables
- YAML config: stream count, model, precision, thresholds, deadline, device.
- Docker image w/ NVIDIA Container Toolkit; compose file with Prometheus + Grafana
  and a dashboard JSON (live latency histogram, queue depth, degradation tier).
- Colab notebook: clone, setup, build engines, run sweep, render charts.
- pytest suite over controller + queue logic using the mock backend. GitHub Actions CI.
- README: architecture, every design decision with rationale, results with clearly
  marked placeholders for my measured numbers.

## Build order — check in with me between stages
1. Skeleton, config system, metrics collection, mock inference backend
2. Single-stream end-to-end with real latency instrumentation
3. Multi-stream: bounded queues, dynamic batching
4. TensorRT integration, precision variants, engine build scripts
5. SLA controller with degradation ladder + hysteresis
6. Experiment automation, charts, Docker, Grafana, README

## Working style
- Explain the approach in plain terms before writing each stage's code.
- Small, scoped commits per stage. No 4000-line initial commit.
- You cannot compile CUDA or run TensorRT here. Write real code, put GPU execution
  behind an interface with a mock implementation so logic is testable locally.
- Keep a running "what didn't work" section in the README — failed optimizations and
  why they failed are the most credible part of this project.