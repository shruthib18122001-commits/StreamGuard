"""Phase 2 / Stage 0: the BEFORE numbers.

Measures the existing host-staged path (CPU decode -> CPU letterbox -> host
NHWC->NCHW float cast -> pinned H2D -> TensorRT -> D2H of the full raw
output -> NumPy YOLOX decode -> NMS) at fixed batch sizes, with no queueing
and no concurrency, so that every later Phase 2 stage has a clean number to
beat. It deliberately does NOT go through MultiStreamPipeline: that pipeline
adds queue wait to total_ms, which is right for the SLA controller but
muddies a per-stage comparison.

Clock choice, per CLAUDE.md: anything that waits on the GPU is timed with
CUDA events (CudaEventTimer on the backend's own stream, plus the event
segments the backend records itself). perf_counter is used only for stages
that run synchronously on the host (CPU decode, cv2 resize, NumPy, NMS),
where there is no asynchronous GPU work for it to misreport.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from streamguard.decode.base import Decoder
from streamguard.inference.base import InferenceBackend
from streamguard.inference.yolox_decode import YOLOX_STRIDES
from streamguard.metrics.timer import Timer, WallClockTimer
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.base import Preprocessor


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """median / p95 / p99 / variance / mean / n -- never a single number."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "variance": float(np.var(arr)),
        "mean": float(np.mean(arr)),
    }


def expected_bytes_per_frame(input_size: int, num_classes: int = 80) -> Dict[str, int]:
    """What the current code SHOULD move per frame, from shapes alone, so the
    measured byte counts can be cross-checked rather than trusted."""
    num_anchors = sum((input_size // s) ** 2 for s in YOLOX_STRIDES)
    return {
        "num_anchors": num_anchors,
        "host_uint8_letterboxed": input_size * input_size * 3,
        "h2d_float32_nchw": input_size * input_size * 3 * 4,
        "d2h_raw_output_float32": num_anchors * (5 + num_classes) * 4,
    }


class _FrameFeeder:
    """Endless frame supply from one clip: reopens at EOF. Decode time is
    taken per read_frame() so the reopen cost never lands in a sample."""

    def __init__(self, decoder_factory: Callable[[], Decoder], source: str):
        self._factory, self._source = decoder_factory, source
        self._decoder: Optional[Decoder] = None
        self.frame_shape = None

    def _reopen(self) -> None:
        if self._decoder is not None:
            self._decoder.close()
        self._decoder = self._factory()
        self._decoder.open(self._source)

    def next(self):
        """Returns (frame, decode_ms)."""
        if self._decoder is None:
            self._reopen()
        t0 = time.perf_counter()  # CPU decode: synchronous host work
        frame = self._decoder.read_frame()
        ms = (time.perf_counter() - t0) * 1000.0
        if frame is None:
            self._reopen()
            t0 = time.perf_counter()
            frame = self._decoder.read_frame()
            ms = (time.perf_counter() - t0) * 1000.0
            if frame is None:
                raise RuntimeError(f"source {self._source!r} yields no frames")
        self.frame_shape = tuple(frame.shape)
        return frame, ms

    def close(self) -> None:
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None


def measure_stage_breakdown(
    decoder_factory: Callable[[], Decoder],
    source: str,
    preprocessor: Preprocessor,
    backend: InferenceBackend,
    postprocessor: Postprocessor,
    input_size: int,
    batch_size: int,
    iterations: int,
    warmup: int,
    inference_timer_factory: Optional[Callable[[], Timer]] = None,
) -> dict:
    """One synchronous batch per iteration: decode B frames, preprocess B,
    one infer(B), postprocess B. Warmup iterations are run and discarded.

    Stage numbers are per BATCH. `total_ms` is their sum (no queue wait in
    this controlled loop); every frame in a batch experiences that whole
    total, so per-frame latency == per-batch total, and per-frame *cost* is
    total / B. Both are reported.
    """
    make_timer = inference_timer_factory or WallClockTimer
    feeder = _FrameFeeder(decoder_factory, source)
    rows: List[dict] = []
    try:
        for it in range(warmup + iterations):
            frames, decode_ms = [], 0.0
            for _ in range(batch_size):
                f, ms = feeder.next()
                frames.append(f)
                decode_ms += ms

            t0 = time.perf_counter()
            processed, metas = [], []
            for f in frames:
                p, m = preprocessor.process(f, input_size)
                processed.append(p)
                metas.append(m)
            preprocess_ms = (time.perf_counter() - t0) * 1000.0

            batch = np.stack(processed, axis=0)
            with make_timer() as t_inf:
                dets = backend.infer(batch)
            inference_ms = t_inf.elapsed_ms

            t0 = time.perf_counter()
            n_final = [len(postprocessor.process(d, m)) for d, m in zip(dets, metas)]
            postprocess_ms = (time.perf_counter() - t0) * 1000.0

            if it < warmup:
                continue
            row = {
                "decode_ms": decode_ms,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": decode_ms + preprocess_ms + inference_ms + postprocess_ms,
                "raw_dets_per_frame": float(np.mean([len(d) for d in dets])),
                "final_dets_per_frame": float(np.mean(n_final)),
            }
            row.update(getattr(backend, "last_timing", {}) or {})
            rows.append(row)
    finally:
        feeder.close()

    stage_keys = ["decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"]
    sub_keys = ["host_prep_ms", "h2d_ms", "exec_ms", "d2h_ms", "host_decode_ms"]
    result = {
        "batch_size": batch_size,
        "input_size": input_size,
        "warmup_discarded": warmup,
        "iterations": len(rows),
        "frame_shape_hwc": list(feeder.frame_shape) if feeder.frame_shape else None,
        "per_batch_ms": {k: summarize([r[k] for r in rows]) for k in stage_keys},
        "per_frame_ms": {k: summarize([r[k] / batch_size for r in rows]) for k in stage_keys},
        "inference_breakdown_per_batch_ms": {k: summarize([r[k] for r in rows]) for k in sub_keys if k in rows[0]},
        "raw_dets_per_frame": summarize([r["raw_dets_per_frame"] for r in rows]),
        "final_dets_per_frame": summarize([r["final_dets_per_frame"] for r in rows]),
        "expected_bytes_per_frame": expected_bytes_per_frame(input_size),
    }
    if "h2d_bytes" in rows[0]:
        result["measured_bytes_per_frame"] = {
            "h2d": rows[0]["h2d_bytes"] / batch_size,
            "d2h": rows[0]["d2h_bytes"] / batch_size,
        }
    return result
