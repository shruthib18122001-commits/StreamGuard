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


@dataclass
class DecodedFrame:
    stream_id: int
    frame: np.ndarray
    decode_ms: float


@dataclass
class PreprocessedItem:
    stream_id: int
    processed_frame: np.ndarray
    meta: PreprocessMeta
    decode_ms: float
    preprocess_ms: float


@dataclass
class InferredItem:
    stream_id: int
    detections: List[Detection]
    meta: PreprocessMeta
    decode_ms: float
    preprocess_ms: float
    inference_ms: float


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
                with WallClockTimer() as t:
                    frame = decoder.read_frame()
                if frame is None:
                    break
                try:
                    self.output_queue.put(
                        DecodedFrame(stream_id=self.stream_id, frame=frame, decode_ms=t.elapsed_ms)
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
        model_input_size: int,
    ):
        super().__init__(name="preprocess")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.preprocessor = preprocessor
        self.model_input_size = model_input_size

    def _run_safe(self) -> None:
        try:
            while True:
                item: Optional[DecodedFrame] = self.input_queue.get()
                if item is None:
                    break
                with WallClockTimer() as t:
                    processed, meta = self.preprocessor.process(item.frame, self.model_input_size)
                try:
                    self.output_queue.put(
                        PreprocessedItem(
                            stream_id=item.stream_id,
                            processed_frame=processed,
                            meta=meta,
                            decode_ms=item.decode_ms,
                            preprocess_ms=t.elapsed_ms,
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
    ):
        super().__init__(name="batch-infer")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.inference_backend = inference_backend
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.batch_sizes_seen: List[int] = []

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

    def _process_batch(self, batch: List[PreprocessedItem]) -> None:
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
                    )
                )
            except QueueClosedError:
                break


class PostprocessWorker(_SafeThread):
    def __init__(self, input_queue: BoundedQueue, postprocessor: Postprocessor, tracker: LatencyTracker):
        super().__init__(name="postprocess")
        self.input_queue = input_queue
        self.postprocessor = postprocessor
        self.tracker = tracker

    def _run_safe(self) -> None:
        while True:
            item: Optional[InferredItem] = self.input_queue.get()
            if item is None:
                break
            with WallClockTimer() as t:
                self.postprocessor.process(item.detections, item.meta)
            total_ms = item.decode_ms + item.preprocess_ms + item.inference_ms + t.elapsed_ms
            self.tracker.record(
                FrameLatency(
                    decode_ms=item.decode_ms,
                    preprocess_ms=item.preprocess_ms,
                    inference_ms=item.inference_ms,
                    postprocess_ms=t.elapsed_ms,
                    total_ms=total_ms,
                    stream_id=item.stream_id,
                    tier=0,  # hardcoded to baseline until the stage-5 SLA controller exists
                    timestamp=time.time(),
                )
            )


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
    ):
        self.decoder_factory = decoder_factory
        self.preprocessor = preprocessor
        self.inference_backend = inference_backend
        self.postprocessor = postprocessor
        self.tracker = tracker
        self.model_input_size = model_input_size
        self.queue_config = queue_config
        self.batch_config = batch_config

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
        preprocess_worker = PreprocessWorker(decode_out_q, batch_in_q, self.preprocessor, self.model_input_size)
        batch_worker = BatchInferenceWorker(
            batch_in_q, post_q, self.inference_backend,
            self.batch_config.max_batch_size, self.batch_config.batch_timeout_ms,
        )
        postprocess_worker = PostprocessWorker(post_q, self.postprocessor, self.tracker)

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
