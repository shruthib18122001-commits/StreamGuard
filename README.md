# StreamGuard

Real-time multi-stream GPU object detection with a hard end-to-end latency
deadline (33ms @ 30fps), enforced by an adaptive degradation controller
rather than by dropping frames uncontrolled. Detection (YOLOX) is commodity;
the SLA controller — degrading quality along a fixed ladder under load, with
hysteresis so it doesn't oscillate — is the point of the project.

Target hardware: developed on macOS, run on Google Colab (T4). Code is kept
Jetson-clean (no x86-only assumptions, no hardcoded TensorRT plugin paths, no
baked-in CUDA arch flags) for a possible later port to Jetson Orin Nano.

## Status

Stage 1 of 6 (skeleton, config, metrics, mock inference backend). See
[Build order](#build-order) below for what's next.

## Architecture

```
Source -> NVDEC decode -> GPU preprocess -> batch -> TensorRT -> postprocess -> out
                                                        ^
                                          SLA Controller (reads p99 latency)
```

- Bounded queues between every stage (backpressure policy TBD — stage 3).
- Decode on GPU via NVDEC, with a CPU decode path kept solely to measure the gap.
- Preprocess on GPU; frames stay device-resident from decode through output.
- Dynamic batching aggregates frames across streams: batch on size N *or* a
  timeout, whichever fires first (stage 3).
- Concurrency model: threads + CUDA streams, one thread per pipeline stage —
  not process-per-stream (separate CUDA contexts would kill cross-stream
  batching) and not asyncio (CUDA work isn't awaitable, and GIL release
  during NVDEC/cudaMemcpyAsync/TRT execute already gives the concurrency
  asyncio would otherwise buy).

## What exists today (stage 1)

- **`streamguard/config.py`** — pydantic config schema (stream count, model
  tier/precision, SLA thresholds, degradation ladder, device) loaded from
  YAML (`configs/default.yaml`). Validates hysteresis sanity
  (`recover_threshold_ms < degrade_threshold_ms`) and ladder ordering at load
  time.

  **Why pydantic over plain dataclasses:** the same validate-at-the-boundary
  model as a FastAPI request body, applied to experiment config — a bad
  threshold fails immediately with a clear message instead of surfacing as
  mysterious controller oscillation three stages later.

- **`streamguard/metrics/latency.py`** — `LatencyTracker`: bounded rolling
  window over per-frame, per-stage latency (decode/preprocess/inference/
  postprocess/total), reporting median/p95/p99/variance. Discards a
  configurable number of warmup samples before anything enters the window.

  **Why it doesn't do any timing itself:** the project's measurement rule is
  CUDA events for GPU stage timing, never `time.time()` — host timing only
  captures kernel *submission*, not completion, since GPU work is
  asynchronous. `LatencyTracker` only aggregates ms values it's given, so
  it's agnostic to how they were measured. The actual CUDA-event
  instrumentation is added in stage 3+, at the call sites that do real GPU
  work.

- **`streamguard/metrics/events.py`** — `TransitionLog` / `TierTransition`:
  the record type for degradation-ladder transitions. Just the log, not the
  decision logic — the controller that decides *when* to transition is
  stage 5.

- **`streamguard/metrics/device.py`** — `capture_device_info()`: best-effort
  snapshot of device / driver version / TensorRT version via `pynvml` /
  `tensorrt`, falling back to `None` when those aren't available (e.g. on
  this Mac dev machine, honestly, rather than faking a value).

- **`streamguard/inference/`** — `InferenceBackend` ABC (`load`, `infer`) and
  `MockInferenceBackend`, which simulates a configurable per-(tier,
  precision) latency and returns randomly-shaped detections. Lets the
  queue/batching engine and the SLA controller be built and unit-tested on
  CPU, with no GPU, before any TensorRT integration exists.

## Degradation ladder (config-defined, controller lands in stage 5)

| Level | Model      | Precision | Size | Frame skip |
|-------|------------|-----------|------|------------|
| 0     | YOLOX-m    | FP16      | 640  | 1          |
| 1     | YOLOX-s    | FP16      | 640  | 1          |
| 2     | YOLOX-s    | INT8      | 640  | 1          |
| 3     | YOLOX-tiny | INT8      | 416  | 1          |
| 4     | YOLOX-nano | INT8      | 416  | 1          |
| 5     | YOLOX-nano | INT8      | 416  | 2          |

## Build order

1. ✅ Skeleton, config system, metrics collection, mock inference backend
2. Single-stream end-to-end with real latency instrumentation
3. Multi-stream: bounded queues, dynamic batching
4. TensorRT integration, precision variants, engine build scripts
5. SLA controller: degradation ladder + hysteresis
6. Experiment automation, charts, Docker, Grafana, README completion

## Results

_Placeholder — populated once real streams run on Colab T4 hardware:_

- Max sustainable stream count per degradation tier: **TBD**
- NVDEC vs CPU decode throughput / CPU utilization: **TBD**
- FP32 / FP16 / INT8 latency, throughput, power, mAP sweep: **TBD**
- Degradation tier transition timeline under load: **TBD**

## What didn't work

_Running log of failed approaches and why — kept because it's often the most
credible part of a systems project. Empty for now; stage 1 has no GPU-facing
code to fail yet._

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```

No GPU required — everything above is testable against the mock inference
backend.
