"""Multi-stream pipeline: N decode threads (one per source — each is an
independent blocking I/O resource) feeding three shared, single-threaded
stages (preprocess, batch+infer, postprocess) over bounded queues.

Cross-stream dynamic batching only makes sense with *one* inference thread
pulling from *one* queue fed by every stream — that's why decode is the
exception to "one thread per pipeline stage," not a violation of it.

Shutdown protocol: each decode thread closes the shared decode-output queue
only once it's the last one still running (via ActiveCounter). Closing a
BoundedQueue lets consumers drain whatever's left, then get() returns None
forever. Each single-threaded stage closes its own output queue when its
input queue reports permanently empty, propagating the shutdown signal one
hop downstream. This is what lets `run()` terminate cleanly once every
source is exhausted, instead of the consumer threads blocking forever.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable, List, Optional

import numpy as np

from streamguard.bounded_queue import BoundedQueue, QueueClosedError
from streamguard.config import BatchConfig, QueueConfig
from streamguard.decode.base import Decoder
from streamguard.inference.base import Detection, InferenceBackend
from streamguard.metrics.latency import FrameLatency, LatencyTracker
from streamguard.metrics.timer import WallClockTimer
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.base import Preprocessor, PreprocessMeta
from streamguard.sla_controller import SLAController


@dataclass
class DecodedFrame:
    stream_id: int
    frame: np.ndarray
    decode_ms: float
    frame_start: float  # perf_counter() at decode start -- see PostprocessWorker


@dataclass
class PreprocessedItem:
    stream_id: int
    processed_frame: np.ndarray
    meta: PreprocessMeta
    decode_ms: float
    preprocess_ms: float
    frame_start: float


@dataclass
class InferredItem:
    stream_id: int
    detections: List[Detection]
    meta: PreprocessMeta
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    frame_start: float
    tier: int = 0


class ActiveCounter:
    """Thread-safe countdown so the last of N decode threads to finish is
    the one that closes their shared output queue."""

    def __init__(self, initial: int):
        self._n = initial
        self._lock = threading.Lock()

    def decrement_and_check_zero(self) -> bool:
        with self._lock:
            self._n -= 1
            return self._n == 0


class _SafeThread(threading.Thread):
    """Captures an exception raised in the thread instead of letting it
    print-and-vanish, so a bug in one stage surfaces as a clear error from
    the orchestrator rather than a silent hang in a downstream thread
    waiting on a queue that never gets closed."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            self._run_safe()
        except BaseException as e:  # noqa: BLE001 - deliberately broad, see docstring
            self.error = e

    def _run_safe(self) -> None:
        raise NotImplementedError


class DecodeWorker(_SafeThread):
    def __init__(
        self,
        stream_id: int,
        source: str,
        decoder_factory: Callable[[], Decoder],
        output_queue: BoundedQueue,
        active_counter: ActiveCounter,
        stop_event: threading.Event,
        max_frames: Optional[int] = None,
    ):
        super().__init__(name=f"decode-{stream_id}")
        self.stream_id = stream_id
        self.source = source
        self.decoder_factory = decoder_factory
        self.output_queue = output_queue
        self.active_counter = active_counter
        self.stop_event = stop_event
        self.max_frames = max_frames
        self.frames_read = 0

    def _run_safe(self) -> None:
        decoder = self.decoder_factory()
        decoder.open(self.source)
        try:
            while not self.stop_event.is_set():
                if self.max_frames is not None and self.frames_read >= self.max_frames:
                    break
                frame_start = time.perf_counter()
                with WallClockTimer() as t:
                    frame = decoder.read_frame()
                if frame is None:
                    break
                try:
                    self.output_queue.put(
                        DecodedFrame(
                            stream_id=self.stream_id, frame=frame, decode_ms=t.elapsed_ms, frame_start=frame_start
                        )
                    )
                except QueueClosedError:
                    break
                self.frames_read += 1
        finally:
            decoder.close()
            if self.active_counter.decrement_and_check_zero():
                self.output_queue.close()


class PreprocessWorker(_SafeThread):
    def __init__(
        self,
        input_queue: BoundedQueue,
        output_queue: BoundedQueue,
        preprocessor: Preprocessor,
        input_size_fn: Callable[[], int],
    ):
        super().__init__(name="preprocess")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.preprocessor = preprocessor
        # A callable, not a fixed int: when an SLAController is active, the
        # target size can change (e.g. 640 -> 416) the moment a degradation
        # tier drops, and every frame must be letterboxed to whatever's
        # active *right now* -- reading it fresh here is what makes that
        # actually take effect, rather than being fixed at construction.
        self.input_size_fn = input_size_fn

    def _run_safe(self) -> None:
        try:
            while True:
                item: Optional[DecodedFrame] = self.input_queue.get()
                if item is None:
                    break
                with WallClockTimer() as t:
                    processed, meta = self.preprocessor.process(item.frame, self.input_size_fn())
                try:
                    self.output_queue.put(
                        PreprocessedItem(
                            stream_id=item.stream_id,
                            processed_frame=processed,
                            meta=meta,
                            decode_ms=item.decode_ms,
                            preprocess_ms=t.elapsed_ms,
                            frame_start=item.frame_start,
                        )
                    )
                except QueueClosedError:
                    break
        finally:
            self.output_queue.close()


class BatchInferenceWorker(_SafeThread):
    def __init__(
        self,
        input_queue: BoundedQueue,
        output_queue: BoundedQueue,
        inference_backend: InferenceBackend,
        max_batch_size: int,
        batch_timeout_ms: float,
        sla_controller: Optional[SLAController] = None,
    ):
        super().__init__(name="batch-infer")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.inference_backend = inference_backend
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.batch_sizes_seen: List[int] = []
        self.sla_controller = sla_controller
        # None, not 0: forces a load() before the very first batch when a
        # controller is present, so this thread's model state is always
        # derived from the controller rather than trusting that whatever
        # the caller pre-loaded happens to match level 0.
        self._loaded_level: Optional[int] = None

    def _collect_batch(self) -> "tuple[List[PreprocessedItem], bool]":
        """Collect until max_batch_size items or batch_timeout_ms since the
        first item of this batch, whichever fires first. The timeout clock
        only starts once the batch has its first item — no point timing out
        a batch that hasn't begun."""
        batch: List[PreprocessedItem] = []
        deadline: Optional[float] = None
        closed = False
        while len(batch) < self.max_batch_size:
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
            try:
                item = self.input_queue.get(timeout=remaining)
            except Empty:
                break
            if item is None:
                closed = True
                break
            batch.append(item)
            if deadline is None:
                deadline = time.perf_counter() + self.batch_timeout_ms / 1000.0
        return batch, closed

    def _run_safe(self) -> None:
        try:
            while True:
                batch, closed = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                if closed:
                    break
        finally:
            self.output_queue.close()

    def _maybe_reload_model(self) -> int:
        """Apply a pending tier change if the controller has moved since our
        last batch. Only this thread ever touches inference_backend, so
        there's no risk of a load() racing an in-flight infer() call."""
        if self.sla_controller is None:
            return 0
        level = self.sla_controller.current_level
        if level != self._loaded_level:
            tier_cfg = self.sla_controller.current_tier_config
            self.inference_backend.load(tier_cfg.model_tier, tier_cfg.precision, tier_cfg.input_size)
            self._loaded_level = level
        return level

    def _process_batch(self, batch: List[PreprocessedItem]) -> None:
        tier = self._maybe_reload_model()
        frames = np.stack([b.processed_frame for b in batch], axis=0)
        with WallClockTimer() as t_infer:
            raw_detections = self.inference_backend.infer(frames)
        self.batch_sizes_seen.append(len(batch))
        # Every frame in this batch is charged the FULL batch latency, not
        # batch_latency / N: all frames in a synchronous batch complete at
        # the same instant, so that's genuinely each one's wait. This is the
        # throughput/latency tradeoff dynamic batching makes explicit.
        for item, dets in zip(batch, raw_detections):
            try:
                self.output_queue.put(
                    InferredItem(
                        stream_id=item.stream_id,
                        detections=dets,
                        meta=item.meta,
                        decode_ms=item.decode_ms,
                        preprocess_ms=item.preprocess_ms,
                        inference_ms=t_infer.elapsed_ms,
                        frame_start=item.frame_start,
                        tier=tier,
                    )
                )
            except QueueClosedError:
                break


class PostprocessWorker(_SafeThread):
    def __init__(
        self,
        input_queue: BoundedQueue,
        postprocessor: Postprocessor,
        tracker: LatencyTracker,
        sla_controller: Optional[SLAController] = None,
    ):
        super().__init__(name="postprocess")
        self.input_queue = input_queue
        self.postprocessor = postprocessor
        self.tracker = tracker
        self.sla_controller = sla_controller

    def _run_safe(self) -> None:
        while True:
            item: Optional[InferredItem] = self.input_queue.get()
            if item is None:
                break
            with WallClockTimer() as t:
                self.postprocessor.process(item.detections, item.meta)
            # True wall-clock elapsed since decode started, NOT the sum of
            # the four stage durations: with real concurrency, a frame can
            # sit in a bounded queue waiting for its turn (e.g. behind a
            # backlog at the batcher) between stages, and that wait is
            # genuinely part of what the SLA controller needs to see. Sum-
            # of-stages only equals this in the queue-free single-stream
            # pipeline; here it would silently under-report latency under
            # exactly the load conditions the controller exists to catch.
            total_ms = (time.perf_counter() - item.frame_start) * 1000.0
            self.tracker.record(
                FrameLatency(
                    decode_ms=item.decode_ms,
                    preprocess_ms=item.preprocess_ms,
                    inference_ms=item.inference_ms,
                    postprocess_ms=t.elapsed_ms,
                    total_ms=total_ms,
                    stream_id=item.stream_id,
                    tier=item.tier,
                    timestamp=time.time(),
                )
            )
            # Right after a fresh sample lands is the natural point to
            # re-check the SLA: the tracker's p99 is as current as it'll get.
            if self.sla_controller is not None:
                self.sla_controller.maybe_transition(self.tracker)


class MultiStreamPipeline:
    def __init__(
        self,
        decoder_factory: Callable[[], Decoder],
        preprocessor: Preprocessor,
        inference_backend: InferenceBackend,
        postprocessor: Postprocessor,
        tracker: LatencyTracker,
        model_input_size: int,
        queue_config: QueueConfig,
        batch_config: BatchConfig,
        sla_controller: Optional[SLAController] = None,
    ):
        self.decoder_factory = decoder_factory
        self.preprocessor = preprocessor
        self.inference_backend = inference_backend
        self.postprocessor = postprocessor
        self.tracker = tracker
        self.model_input_size = model_input_size
        self.queue_config = queue_config
        self.batch_config = batch_config
        # Optional: without one, every frame stays tagged tier=0 and the
        # model is never reloaded mid-run -- identical to pre-stage-5
        # behavior. With one, PreprocessWorker reads its target size from
        # the controller every frame, and BatchInferenceWorker reloads the
        # model (and tags each frame's real tier) whenever it's moved.
        self.sla_controller = sla_controller

    def run(
        self,
        sources: List[str],
        max_frames_per_stream: Optional[int] = None,
        join_timeout: Optional[float] = 30.0,
    ) -> dict:
        decode_out_q = BoundedQueue(self.queue_config.decode_queue_size, self.queue_config.decode_queue_policy)
        batch_in_q = BoundedQueue(self.queue_config.batch_queue_size, self.queue_config.batch_queue_policy)
        post_q = BoundedQueue(self.queue_config.postprocess_queue_size, self.queue_config.postprocess_queue_policy)

        active = ActiveCounter(len(sources))
        stop_event = threading.Event()

        decode_workers = [
            DecodeWorker(i, src, self.decoder_factory, decode_out_q, active, stop_event, max_frames_per_stream)
            for i, src in enumerate(sources)
        ]
        if self.sla_controller is not None:
            input_size_fn = lambda: self.sla_controller.current_tier_config.input_size  # noqa: E731
        else:
            fixed_size = self.model_input_size
            input_size_fn = lambda: fixed_size  # noqa: E731
        preprocess_worker = PreprocessWorker(decode_out_q, batch_in_q, self.preprocessor, input_size_fn)
        batch_worker = BatchInferenceWorker(
            batch_in_q, post_q, self.inference_backend,
            self.batch_config.max_batch_size, self.batch_config.batch_timeout_ms,
            sla_controller=self.sla_controller,
        )
        postprocess_worker = PostprocessWorker(post_q, self.postprocessor, self.tracker, sla_controller=self.sla_controller)

        all_workers: List[_SafeThread] = [*decode_workers, preprocess_worker, batch_worker, postprocess_worker]
        for w in all_workers:
            w.start()
        for w in all_workers:
            w.join(timeout=join_timeout)

        hung = [w.name for w in all_workers if w.is_alive()]
        if hung:
            stop_event.set()
            raise RuntimeError(f"pipeline workers did not shut down within {join_timeout}s: {hung}")

        errors = [(w.name, w.error) for w in all_workers if w.error is not None]
        if errors:
            raise RuntimeError(f"pipeline worker(s) failed: {errors}") from errors[0][1]

        return {
            "frames_processed": self.tracker.total_seen,
            "dropped_decode_queue": decode_out_q.dropped_count,
            "dropped_batch_queue": batch_in_q.dropped_count,
            "batch_sizes_seen": list(batch_worker.batch_sizes_seen),
        }
