#!/usr/bin/env python3
"""Phase 2 Stage 0: record the BEFORE numbers on real hardware.

One run produces results/stage0_baseline.json containing, for one engine:
  - device / driver / CUDA / TensorRT / git commit
  - a sanity check that the engine produces plausible detections on a real frame
  - per-stage latency (decode / preprocess / inference / postprocess / total)
    at each requested batch size, as median / p95 / p99 / variance
  - the inference stage split into host_prep / H2D / exec / D2H / host_decode
  - bytes moved H2D and D2H per frame (measured, next to the shape-derived value)
  - the real SingleStreamPipeline run with CudaEventTimer on the inference stage
  - NVML telemetry over the run (clocks, power, temperature, throttle reasons)
  - one time-boxed NVDEC attempt (success or the exact error)

Usage (Colab):
    python scripts/run_baseline.py --engine-dir engines --tier tiny --precision fp16 \
        --input-size 416 --video /content/sample_videos/example.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter

from streamguard.config import ModelTier, Precision
from streamguard.decode.cpu import CPUDecoder
from streamguard.experiments.baseline import measure_stage_breakdown
from streamguard.metrics.device import capture_device_info
from streamguard.metrics.latency import LatencyTracker
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.cpu import CPUPreprocessor


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def _lib_versions() -> dict:
    v = {"python": platform.python_version()}
    for name in ("numpy", "cv2", "cupy", "tensorrt"):
        try:
            v[name] = __import__(name).__version__
        except Exception as e:  # noqa: BLE001
            v[name] = f"unavailable ({type(e).__name__})"
    return v


def _nvml_summary(samples) -> dict:
    if not samples:
        return {"available": False}
    def col(attr):
        return [getattr(s, attr) for s in samples if getattr(s, attr) is not None]
    def rng(vals):
        return {"min": min(vals), "max": max(vals)} if vals else None
    reasons = Counter(r for s in samples for r in s.throttle_reasons)
    return {
        "available": True,
        "n_samples": len(samples),
        "power_watts": rng(col("power_watts")),
        "sm_clock_mhz": rng(col("sm_clock_mhz")),
        "memory_clock_mhz": rng(col("memory_clock_mhz")),
        "temperature_c": rng(col("temperature_c")),
        "memory_used_mb": rng(col("memory_used_mb")),
        "throttle_reason_sample_counts": dict(reasons),
    }


def _nvdec_child(video: str) -> None:
    """Runs in a subprocess so a native crash in the binding cannot kill the
    baseline run. Prints one JSON line on success."""
    from streamguard.decode.nvdec import NVDECDecoder

    dec = NVDECDecoder()
    dec.open(video)
    n, t0 = 0, time.perf_counter()
    while n < 120 and dec.read_frame() is not None:
        n += 1
    dt = time.perf_counter() - t0
    dec.close()
    print(json.dumps({"frames": n, "seconds": dt, "fps": n / dt if dt else None}))


def _nvdec_probe(video: str, timeout_s: int = 180) -> dict:
    """ONE attempt, per the Stage 0 time-box. Failure is a recorded result."""
    try:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--_nvdec-child", video],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout_s}s"}
    if p.returncode == 0 and p.stdout.strip():
        return {"ok": True, **json.loads(p.stdout.strip().splitlines()[-1])}
    return {"ok": False, "returncode": p.returncode, "stderr_tail": p.stderr[-2000:], "stdout_tail": p.stdout[-500:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-dir", default="engines")
    ap.add_argument("--tier", choices=[t.value for t in ModelTier], default="tiny")
    ap.add_argument("--precision", choices=[p.value for p in Precision], default="fp16")
    ap.add_argument("--input-size", type=int, default=416)
    ap.add_argument("--video", default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--single-stream-frames", type=int, default=300)
    ap.add_argument("--output", default="results/stage0_baseline.json")
    ap.add_argument("--skip-nvdec", action="store_true")
    ap.add_argument("--_nvdec-child", dest="nvdec_child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.nvdec_child:
        _nvdec_child(args.nvdec_child)
        return

    if not args.video:
        ap.error("--video is required")

    from streamguard.inference.tensorrt_backend import TensorRTInferenceBackend
    from streamguard.metrics.cuda_timer import CudaEventTimer
    from streamguard.metrics.nvml_sampler import NVMLSampler
    from streamguard.pipeline import SingleStreamPipeline

    info = capture_device_info("cuda:0")
    assert info.driver_version, "no NVIDIA driver visible -- is this a GPU runtime?"
    out = {
        "stage": "phase2-stage0-baseline",
        "timestamp": time.time(),
        "git_commit": _git_commit(),
        "device": info.__dict__,
        "libraries": _lib_versions(),
        "config": {k: getattr(args, k) for k in ("tier", "precision", "input_size", "batch_sizes", "iterations", "warmup", "video")},
        "path_description": "CPU decode (cv2) -> cv2 letterbox -> host NHWC uint8 -> NCHW float32 into pinned -> H2D -> TRT -> "
                            "D2H full raw output -> NumPy YOLOX decode -> NumPy NMS",
    }

    backend = TensorRTInferenceBackend(engine_dir=args.engine_dir)
    backend.load(ModelTier(args.tier), Precision(args.precision), args.input_size)
    event_timer = lambda: CudaEventTimer(stream=backend.stream)  # noqa: E731 - events must be on the backend's stream

    sampler = NVMLSampler(interval_s=0.5)
    sampler.start()
    try:
        # --- sanity: does the engine see anything in a real frame? ---
        dec = CPUDecoder(); dec.open(args.video)
        frame = None
        for _ in range(30):  # skip a few frames; first ones are often black
            frame = dec.read_frame()
        dec.close()
        if frame is None:
            raise RuntimeError(f"could not read a frame from {args.video!r}")
        proc, meta = CPUPreprocessor().process(frame, args.input_size)
        dets = backend.infer(proc[None])[0]
        final = Postprocessor().process(dets, meta)
        out["sanity"] = {
            "frame_shape_hwc": list(frame.shape),
            "raw_detections_over_threshold": len(dets),
            "after_nms": len(final),
            "top5": [{"class_id": d.class_id, "conf": round(d.confidence, 3),
                      "box_xyxy": [round(d.x1), round(d.y1), round(d.x2), round(d.y2)]}
                     for d in sorted(final, key=lambda d: -d.confidence)[:5]],
        }
        print("sanity:", json.dumps(out["sanity"], indent=1))
        if not final:
            print("WARNING: zero detections on a real frame -- engine or layout is probably wrong; "
                  "timings below are not trustworthy until this is explained.")

        # --- per-stage breakdown at each batch size ---
        out["batches"] = []
        for b in args.batch_sizes:
            r = measure_stage_breakdown(
                CPUDecoder, args.video, CPUPreprocessor(), backend, Postprocessor(),
                input_size=args.input_size, batch_size=b, iterations=args.iterations, warmup=args.warmup,
                inference_timer_factory=event_timer,
            )
            out["batches"].append(r)
            pf = r["per_frame_ms"]
            print(f"batch {b}: per-batch total median {r['per_batch_ms']['total_ms']['median']:.2f} ms, "
                  f"inference {r['per_batch_ms']['inference_ms']['median']:.2f} ms, "
                  f"H2D {r['measured_bytes_per_frame']['h2d']/1e6:.2f} MB/frame, "
                  f"D2H {r['measured_bytes_per_frame']['d2h']/1e6:.2f} MB/frame")

        # --- the real single-stream pipeline, inference stage under CudaEventTimer ---
        tracker = LatencyTracker(window_size=args.single_stream_frames, warmup_samples=30)
        pipe = SingleStreamPipeline(
            decoder=CPUDecoder(), preprocessor=CPUPreprocessor(), inference_backend=backend,
            postprocessor=Postprocessor(), tracker=tracker, model_input_size=args.input_size,
            inference_timer_factory=event_timer,
        )
        n = pipe.run(args.video, max_frames=args.single_stream_frames + 30)
        out["single_stream_pipeline"] = {"frames": n, "stage_breakdown_ms": tracker.stage_breakdown()}
    finally:
        sampler.stop()
    out["nvml"] = {"unavailable_reason": sampler.unavailable_reason, **_nvml_summary(sampler.samples)}

    out["nvdec_attempt"] = {"skipped": True} if args.skip_nvdec else _nvdec_probe(args.video)
    print("nvdec:", json.dumps(out["nvdec_attempt"])[:600])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
