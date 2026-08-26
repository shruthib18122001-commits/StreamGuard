"""Per-stage timing, behind a swappable interface.

WallClockTimer (perf_counter) is legitimate for CPU-only stages — the
CUDA-events-not-time.time() rule exists because host timing only captures
async GPU kernel *submission*, not completion. There's no async GPU kernel
here to be fooled by yet. A CudaEventTimer implementing this same context-
manager contract lands once a stage actually submits GPU work (real
TensorRT inference in stage 4, or NVDEC/CuPy decode/preprocess once running
on Colab) — the pipeline code that does `with timer:` around a stage won't
need to change, only which Timer it's given.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class Timer(ABC):
    @abstractmethod
    def __enter__(self) -> "Timer":
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        ...

    @property
    @abstractmethod
    def elapsed_ms(self) -> float:
        ...


class WallClockTimer(Timer):
    def __enter__(self) -> "WallClockTimer":
        self._start = time.perf_counter()
        self._elapsed_ms = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False

    @property
    def elapsed_ms(self) -> float:
        if self._elapsed_ms is None:
            raise RuntimeError("elapsed_ms read before the timer's `with` block exited")
        return self._elapsed_ms
