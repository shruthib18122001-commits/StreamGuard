"""Single-stream, single-threaded pipeline: decode -> preprocess -> infer ->
postprocess -> record latency. No queues, no batching across streams — that's
stage 3's MultiStreamPipeline. This exists to get the per-stage interfaces
and instrumentation right on the simplest possible case.

Optionally SLA-controller-aware (stage 5): with no controller, behavior is
identical to before (tier always 0, model never reloaded). With one, this is
much simpler than the multi-threaded pipeline's wiring -- there's only one
thread, so there's no "which thread is allowed to call load()" question.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from streamguard.decode.base import Decoder
from streamguard.inference.base import Detection, InferenceBackend
from streamguard.metrics.latency import FrameLatency, LatencyTracker
from streamguard.metrics.timer import WallClockTimer
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.base import Preprocessor
from streamguard.sla_controller import SLAController


class SingleStreamPipeline:
    def __init__(
        self,
        decoder: Decoder,
        preprocessor: Preprocessor,
        inference_backend: InferenceBackend,
        postprocessor: Postprocessor,
        tracker: LatencyTracker,
        model_input_size: int,
        stream_id: int = 0,
        sla_controller: Optional[SLAController] = None,
    ):
        self.decoder = decoder
        self.preprocessor = preprocessor
        self.inference_backend = inference_backend
        self.postprocessor = postprocessor
        self.tracker = tracker
        self.model_input_size = model_input_size
        self.stream_id = stream_id
        self.sla_controller = sla_controller
        self._loaded_level: Optional[int] = None

    def _current_input_size(self) -> int:
        if self.sla_controller is None:
            return self.model_input_size
        return self.sla_controller.current_tier_config.input_size

    def _maybe_reload_model(self) -> int:
        if self.sla_controller is None:
            return 0
        level = self.sla_controller.current_level
        if level != self._loaded_level:
            tier_cfg = self.sla_controller.current_tier_config
            self.inference_backend.load(tier_cfg.model_tier, tier_cfg.precision, tier_cfg.input_size)
            self._loaded_level = level
        return level

    def run(self, source: str, max_frames: Optional[int] = None) -> int:
        """Run the pipeline over `source` until it's exhausted or
        `max_frames` is reached. Returns the number of frames processed."""
        self.decoder.open(source)
        frame_idx = 0
        try:
            while max_frames is None or frame_idx < max_frames:
                with WallClockTimer() as t_decode:
                    frame = self.decoder.read_frame()
                if frame is None:
                    break

                with WallClockTimer() as t_pre:
                    processed, meta = self.preprocessor.process(frame, self._current_input_size())

                tier = self._maybe_reload_model()
                batch = processed[np.newaxis, ...]
                with WallClockTimer() as t_infer:
                    raw_detections: List[List[Detection]] = self.inference_backend.infer(batch)

                with WallClockTimer() as t_post:
                    self.postprocessor.process(raw_detections[0], meta)

                total_ms = t_decode.elapsed_ms + t_pre.elapsed_ms + t_infer.elapsed_ms + t_post.elapsed_ms
                self.tracker.record(
                    FrameLatency(
                        decode_ms=t_decode.elapsed_ms,
                        preprocess_ms=t_pre.elapsed_ms,
                        inference_ms=t_infer.elapsed_ms,
                        postprocess_ms=t_post.elapsed_ms,
                        total_ms=total_ms,
                        stream_id=self.stream_id,
                        tier=tier,
                        timestamp=time.time(),
                    )
                )
                if self.sla_controller is not None:
                    self.sla_controller.maybe_transition(self.tracker)
                frame_idx += 1
        finally:
            self.decoder.close()
        return frame_idx
