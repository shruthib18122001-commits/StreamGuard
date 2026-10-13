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
  Phase 2 replaces the CuPy ops with a fused CUDA kernel that writes straight into the
  TensorRT input tensor (see "Phase 2: custom GPU kernels" below).
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

## Phase 1 build order (complete; GPU-dependent parts still unverified on real hardware)
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

## Phase 2: custom GPU kernels

Goal: replace the most data-heavy host and CuPy steps with hand-written CUDA kernels, prove each one correct against a reference, and measure what it buys with CUDA events and bandwidth math. The headline result stays the project's own metric: max sustainable stream count per tier, before vs after.

### What the current code does (verify on Colab before building on it; delete or update this subsection once Stage 0 confirms or refutes each point)
- The pipeline is not actually device-resident. multi_stream_pipeline.py `_process_batch` does np.stack on host arrays. tensorrt_backend.infer() then does batch.astype(float32) on the host (4x the bytes of uint8), copies into pinned memory and H2D copies it. GPUPreprocessor (CuPy) returns a CuPy array that this path cannot consume.
- After inference, the full raw output (batch x 8400 x 85 float32 at 640px, about 2.9 MB per frame) is copied D2H and decoded in NumPy inside the single inference thread, which is on the critical path.
- scripts/build_engine.py sets the optimization profile as NHWC (N,H,W,3) and the backend allocates NHWC buffers. Standard YOLOX ONNX exports are NCHW (N,3,H,W). Expect the first engine build or run to break. Confirm and fix in Stage 0.
- preprocess/gpu.py uses cupyx zoom, whose default coordinate mapping probably differs from cv2.INTER_LINEAR's half-pixel convention, so the docstring's claim that it matches CPUPreprocessor exactly is unproven. The new kernel must match cv2 explicitly and a test must prove it.

### Operating rules (Phase 2)
- All existing rules above still apply: CUDA events never time.time(), discard warmup, report median, p95, p99 and variance, NEVER invent numbers, raw JSON in results/ with device, driver, CUDA, TensorRT and git commit.
- Work in stages. STOP after each stage with a summary of what was built, what was measured and what is next. Do not start the next stage until I say go.
- Write kernels in CUDA C via CuPy RawKernel (NVRTC, so no nvcc is needed on Colab and no new dependency). Compile for the detected device, no hardcoded arch flags (Jetson-clean).
- Every kernel sits behind an interface with a NumPy reference implementation. Tests compare kernel vs reference with explicit tolerances. GPU tests auto-skip when CUDA is unavailable. CI stays CPU-only.
- Before writing each kernel, explain in plain terms: the thread and block mapping, the memory access pattern and whether it coalesces, exactly which bytes are read and written, and whether the kernel is memory-bound or compute-bound.
- Roofline: for every kernel report achieved GB/s against (a) the T4 spec bandwidth (look it up and cite the source, do not rely on memory) and (b) a measured device-to-device copy bandwidth on the same machine.
- Try Nsight Systems or Nsight Compute on Colab. If profiler counters are blocked, say so and rely on CUDA events plus bandwidth math. Never fabricate profiler output.
- If something fails on T4, show me the exact error and the options. Do not work around it silently.
- Add every failed attempt and every kernel that bought little to the README's "What didn't work" section, with the reason.

### Stages
0. Baseline on real hardware. Run the existing Colab runbook end to end on a T4: install, confirm GPU, driver and TensorRT versions, get a YOLOX nano or tiny ONNX, build an FP16 engine, run TensorRTInferenceBackend on one batch, run the single-stream pipeline with CudaEventTimer. Fix what breaks (expect the NHWC vs NCHW problem; NVDEC is the least verified file, so time-box it to one attempt and if it fails document it and continue with CPU decode). Record the BEFORE numbers: per-stage latency (decode, preprocess, inference, postprocess, total) at batch 1 and 8, bytes moved H2D and D2H per frame, and max sustainable stream count at the baseline tier if the sweep runs. Save to results/ as JSON and fill the README Results section with these real numbers. Write no kernel until this stage is done.
1. Fused preprocess kernel. One kernel replaces cupyx zoom, the canvas fill and copy, the host cast to float, and the HWC to CHW conversion. Input: one decoded uint8 HWC frame on the device, any resolution. Output: written directly into slot i of a preallocated device batch tensor, NCHW, in the dtype and value range the engine expects (record what YOLOX expects and cite it). Same letterbox math as PreprocessMeta (scale, pad_x, pad_y), gray-114 padding, bilinear with cv2's half-pixel convention. Tests against CPUPreprocessor plus a NumPy transpose and cast, over odd sizes and aspect ratios, with an explicit tolerance (cv2 uses fixed-point arithmetic on uint8, so expect off-by-one differences and state the tolerance). Benchmark 720p, 1080p and 4K inputs to 416 and 640 targets at batch 1, 8 and 16: cv2 plus host cast plus H2D, vs the current CuPy path, vs the fused kernel. Report ms (CUDA events) and achieved GB/s.
2. Device-resident batch path. The preprocessor writes into the backend's input tensor, with no np.stack and no host astype. Add an infer_device(d_input) entry point to the TensorRT backend and keep the host path for CPU decode and the mock backend. All existing tests must still pass. Add a test that proves batch assembly is slot-addressed (each slot gets the right frame). Measure bytes moved and per-stage latency against Stage 0.
3. Fused YOLOX decode and compaction kernel. Over (batch x anchors): decode the box from grid and stride, take the argmax over classes (parallel reduction across the class dimension), compute conf = obj * cls, threshold, and compact survivors into a fixed-capacity per-frame buffer. Pick atomic-counter compaction or prefix-sum compaction and explain the tradeoff (atomics give nondeterministic order). Only the survivor count and the compacted boxes go D2H. The capacity limit and the overflow behaviour must be explicit and tested. Keep NMS on the host for this stage. Tests: same detection set as decode_yolox_output (order-insensitive compare), including the overflow case. Benchmark decode ms and D2H bytes per frame vs the NumPy path. Stretch: GPU NMS with an IoU bitmask.
4. Re-run the experiments. Re-run the stream-count sweep and precision sweep with the new path. Report max sustainable stream count per tier and the per-stage breakdown, before vs after, with raw JSON. If a kernel bought little or nothing, say so and explain why. Update the charts.
5. Stretch (only if I ask): port the fused preprocess kernel to Triton with the same tests. Compare performance and code complexity vs CUDA C, with one paragraph on when each is the right tool.
6. README. Add a "Custom kernels" section: a plain-language explanation of each kernel, the roofline table, the before/after table, what didn't work, and limitations (single T4, YOLOX only, no attention or GEMM kernels).