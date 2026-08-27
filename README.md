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

Stage 4 of 6 (TensorRT integration, precision variants, engine build
scripts). See [Build order](#build-order) below for what's next, and
[GPU-only code](#gpu-only-code-unverified-until-colab) for what still needs
validating on real hardware.

## Architecture

```
Source -> NVDEC decode -> GPU preprocess -> batch -> TensorRT -> postprocess -> out
                                                        ^
                                          SLA Controller (reads p99 latency)
```

- Bounded queues between every stage. Pre-inference queues (decode→preprocess,
  preprocess→batch) drop the oldest frame under backpressure — a stale frame
  only burns deadline budget, so skipping ahead to a fresher one is strictly
  better. The post-inference queue blocks instead: by then real (simulated)
  GPU time has already been spent, so throwing away a completed result is
  pure waste.
- Decode on GPU via NVDEC, with a CPU decode path kept solely to measure the gap.
- Preprocess on GPU; frames stay device-resident from decode through output.
- Dynamic batching aggregates frames across streams: batch on size N *or* a
  timeout since the first frame of the batch arrived, whichever fires first.
- Concurrency model: threads + CUDA streams, one thread per pipeline stage —
  **except decode**, which gets one thread *per stream* (each source is an
  independent blocking I/O resource). Preprocess, batch+infer, and
  postprocess stay single shared threads, which is exactly what makes
  cross-stream batching possible: one inference thread pulling from one
  queue fed by every stream. Not process-per-stream (separate CUDA contexts
  would kill cross-stream batching) and not asyncio (CUDA work isn't
  awaitable, and GIL release during NVDEC/cudaMemcpyAsync/TRT execute
  already gives the concurrency asyncio would otherwise buy).

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

## What exists today (stage 2 additions)

- **`streamguard/decode/`** — `Decoder` ABC + `CPUDecoder` (OpenCV
  `VideoCapture`). Real decode, testable locally. `NVDECDecoder` is
  deliberately not written yet — there's no NVIDIA driver on this dev
  machine to verify a binding against, and unlike TensorRT (unavoidable,
  landing stage 4 regardless), GPU-vs-CPU decode is an *experiment* to run
  later, not a stage-2 requirement. It arrives once we're actually on Colab
  and can pick a library against real hardware instead of guessing blind.

- **`streamguard/preprocess/`** — `Preprocessor` ABC + `CPUPreprocessor`:
  standard YOLOX-style letterbox resize (aspect-preserving, gray-114 pad —
  a training-convention value, not arbitrary). Returns `PreprocessMeta`
  (scale + pad) alongside the resized frame so postprocessing can map boxes
  back to the original frame's coordinates. Output stays `(H, W, C)` to
  match `InferenceBackend.infer`'s documented batch shape — no premature
  NCHW/normalization conversion; that's a specific TensorRT engine's input
  contract, decided in stage 4, not a general preprocessing concern.

- **`streamguard/postprocess.py`** — `Postprocessor`: confidence
  thresholding, vectorized greedy NMS, then inverts the letterbox transform
  back to original-frame coordinates. This is real, tested logic (not
  mocked) — it doesn't care whether the boxes it receives came from the
  mock backend or a real TensorRT engine.

- **`streamguard/metrics/timer.py`** — `Timer` ABC + `WallClockTimer`
  (`time.perf_counter`, used via a `with` block around each stage).
  Legitimate for CPU-only stages: the CUDA-events-not-`time.time()` rule
  exists specifically because host timing only captures async GPU kernel
  *submission*, not completion — there's no async GPU kernel here yet to be
  fooled by. A `CudaEventTimer` implementing the same context-manager
  contract lands once a stage actually submits GPU work (stage 4
  inference, or NVDEC/CuPy once running on Colab); the pipeline code that
  does `with timer:` won't need to change, only which `Timer` it's given.

- **`streamguard/pipeline.py`** — `SingleStreamPipeline`: wires decode →
  preprocess → infer → postprocess into one sequential loop for a single
  stream, feeding every frame's per-stage timing into stage 1's
  `LatencyTracker`. No queues, no threads, no batching across streams, no
  SLA controller yet — those are stages 3 and 5. This stage exists purely
  to get the interfaces and instrumentation right on the simplest possible
  case before concurrency is layered on top.

## What exists today (stage 3 additions)

- **`streamguard/bounded_queue.py`** — `BoundedQueue`: thread-safe, with an
  explicit per-instance backpressure policy. `BLOCK` behaves like stdlib
  `queue.Queue`; `DROP_OLDEST` evicts the oldest item to make room instead
  of blocking the producer, and counts every eviction (`dropped_count`) —
  the stdlib queue has no such mode, so this wraps a `deque` with our own
  locking to add one. `get(timeout=...)` raises `queue.Empty` on timeout,
  and returns `None` — a distinct, permanent sentinel — only once the queue
  has been closed *and* drained.

- **`streamguard/config.py`** additions — `QueueConfig` (per-hop size +
  `BackpressurePolicy`) and `BatchConfig` (`max_batch_size`,
  `batch_timeout_ms`). Defaults match the policy above: drop-oldest pre-
  inference, block post-inference.

- **`streamguard/multi_stream_pipeline.py`** — `MultiStreamPipeline` and its
  four worker types (`DecodeWorker` ×N, `PreprocessWorker`,
  `BatchInferenceWorker`, `PostprocessWorker`), connected by `BoundedQueue`s.
  `BatchInferenceWorker` collects frames from any stream until
  `max_batch_size` is reached or `batch_timeout_ms` has elapsed since the
  first frame of the current batch, whichever fires first, then runs one
  `inference_backend.infer()` call per batch.

  **A subtlety worth flagging:** every frame in a dispatched batch is
  charged the *full* batch inference latency, not `latency / N` — all
  frames in a synchronous batch complete at the same instant, so that's
  genuinely each one's wait. Measured with 4 concurrent synthetic streams
  and a fixed ~25ms mock inference cost per call: `max_batch_size=1` gave
  34 fps at p99 total latency 33.1ms (right at the deadline); `max_batch_size=8`
  gave 238 fps — a ~7x throughput gain — but p99 rose to 38.1ms, over the
  33ms deadline. Bigger batches move more frames through the GPU per
  second, at the cost of every one of those frames waiting longer. This is
  exactly the tension the stage-5 SLA controller has to manage.

  Each worker thread captures its own exceptions instead of dying silently
  (`_SafeThread`), and `MultiStreamPipeline.run()` checks for both hung
  threads (via a join timeout) and captured errors after shutdown — with N
  decode threads, a shared batching stage, and bounded queues in between,
  a bug that silently wedges one thread would otherwise surface as a
  mysterious test hang instead of a clear failure.

## What exists today (stage 4 additions)

- **`streamguard/inference/yolox_decode.py`** — `decode_yolox_output()`:
  turns YOLOX's raw anchor-free network output (grid-cell predictions
  across feature-map strides 8/16/32) into the same `Detection` objects the
  mock backend already returns, so `postprocess.py`'s NMS and coordinate
  inversion don't change at all regardless of which backend produced them.
  Pure NumPy math on tensor shapes — **this is the one piece of stage 4
  actually verified here**, including an exact-coordinate test (a
  hand-encoded anchor decodes to precisely the expected box).

- **`streamguard/metrics/cuda_timer.py`** — `CudaEventTimer`, implementing
  the same `Timer` contract as `WallClockTimer` (stage 2) but using real
  CUDA events instead of a host clock — this is the actual mechanism
  CLAUDE.md's measurement rule requires for GPU stages. **Unverified**: no
  CuPy, no CUDA device here.

- **`streamguard/inference/tensorrt_backend.py`** — `TensorRTInferenceBackend`,
  implementing the same `InferenceBackend` interface the mock backend does
  (loads a serialized engine, one execution context per instance, pinned
  host buffers for the H2D copy, calls the decode function above). Uses
  CuPy — not `pycuda` — for all device memory, since CuPy's already the
  intended tool for GPU preprocessing. **Unverified**: written against the
  documented TensorRT 8.5+ `execute_async_v3`/tensor-address API, but there
  is no NVIDIA driver, CUDA, or TensorRT on this dev machine to actually
  run it against.

- **`scripts/build_engine.py`** — ONNX → `.engine`, precision selectable
  (fp32/fp16/int8), with an `IInt8EntropyCalibrator2` for the INT8 path
  that reads a folder of representative frames (path supplied externally;
  never committed, per `.gitignore`). Deliberately does **not** handle
  YOLOX-checkpoint → ONNX export — that needs YOLOX's own
  `tools/export_onnx.py` and its model-definition code, and pulling that
  dependency into this repo just to re-derive ONNX export was the wrong
  trade. This script starts from an already-exported `.onnx` file.
  **Unverified**, same reason as above.

- Three tests (`test_gpu_backends_fail_clearly.py`) lock in a narrower,
  honestly-testable promise for the two GPU-only modules: importing them
  must not require CuPy/TensorRT (their imports are deferred into method
  bodies, not module level), and calling into them without their
  prerequisites must fail with a specific, clear error — not silently
  produce garbage or die three layers down with an obscure traceback.

### GPU-only code: unverified until Colab

Everything above marked unverified needs to be checked against real
hardware before it's trusted, in this order:

1. `python -c "import tensorrt, cupy; print(tensorrt.__version__, cupy.__version__)"`
   — confirm the libraries this code assumes are actually what's installed.
2. `capture_device_info()` (stage 1) should return a real driver version and
   TensorRT version, not `None` — if it's still `None` on Colab, something's
   wrong with the environment, not the code.
3. Export one YOLOX checkpoint to ONNX (YOLOX's own tooling), then build the
   cheapest engine first: `scripts/build_engine.py --tier nano --precision
   int8 --input-size 416 --calib-dir <folder>`. Confirm it produces a
   `.engine` file without error.
4. Load that engine with `TensorRTInferenceBackend` and call `infer()` on one
   real preprocessed frame. Confirm no exceptions and sane detection
   count/box coordinates.
5. Swap `TensorRTInferenceBackend` in for `MockInferenceBackend` in
   `SingleStreamPipeline` against one short real video before touching
   `MultiStreamPipeline` — isolate TensorRT problems from concurrency
   problems.
6. Only then swap `CudaEventTimer` in for `WallClockTimer` around the
   inference stage, and confirm the CUDA-event-measured latency is stable
   across repeated runs.

The full precision/stream-count sweep is stage 6 — this checklist is just
"does the new code work at all," not the experiment automation itself.

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
2. ✅ Single-stream end-to-end with real latency instrumentation
3. ✅ Multi-stream: bounded queues, dynamic batching
4. ✅ TensorRT integration, precision variants, engine build scripts (unverified — see checklist above)
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

No GPU required. Every stage through 3 is fully tested against the mock
inference backend; stage 4's GPU-only modules (`tensorrt_backend.py`,
`cuda_timer.py`, `scripts/build_engine.py`) are only tested for graceful,
clear failure in the absence of CUDA/TensorRT — see
[GPU-only code](#gpu-only-code-unverified-until-colab) above for what
actually validates them.
