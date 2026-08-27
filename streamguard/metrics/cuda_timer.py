"""CudaEventTimer: real CUDA-events-based stage timing.

Implements the same `Timer` contract `WallClockTimer` (stage 2) does, so
pipeline code that does `with timer:` around a stage never changes -- only
which Timer instance it's given. This is what CLAUDE.md's measurement rule
actually requires for GPU stages: host timing only captures kernel
*submission*, not completion, since GPU work is asynchronous; CUDA events
are recorded on the stream itself and `synchronize()` genuinely blocks
until the recorded point is reached, so elapsed time reflects real
completion.

Cannot be exercised on this dev machine: needs CuPy and an actual CUDA
device. The cupy import is deferred into __init__ (not at module level) so
this module stays importable -- for tests, type hints, wiring code -- on a
machine with neither.
"""
from __future__ import annotations

from typing import Any, Optional

from streamguard.metrics.timer import Timer


class CudaEventTimer(Timer):
    def __init__(self, stream: Optional[Any] = None):
        import cupy as cp  # deferred: requires CUDA -- see module docstring

        self._cp = cp
        self._stream = stream if stream is not None else cp.cuda.Stream.null
        self._start_event = cp.cuda.Event()
        self._end_event = cp.cuda.Event()
        self._elapsed_ms: Optional[float] = None

    def __enter__(self) -> "CudaEventTimer":
        self._elapsed_ms = None
        self._start_event.record(self._stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._end_event.record(self._stream)
        self._end_event.synchronize()  # block the host until the GPU work actually completes
        self._elapsed_ms = self._cp.cuda.get_elapsed_time(self._start_event, self._end_event)
        return False

    @property
    def elapsed_ms(self) -> float:
        if self._elapsed_ms is None:
            raise RuntimeError("elapsed_ms read before the timer's `with` block exited")
        return self._elapsed_ms
