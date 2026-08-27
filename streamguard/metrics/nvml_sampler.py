"""Continuous NVML sampling: power, SM/memory clocks, temperature, throttle
reasons, memory usage -- sampled on a background thread at a fixed
interval while a real run is in progress. This is what tells "code got
slower" apart from "GPU throttled," which a one-time device/driver
snapshot (metrics/device.py) can't do.

Same graceful-degradation pattern as device.py: on a machine with no NVIDIA
driver (this one), the sampler starts, runs, and stops cleanly, producing
zero samples and one clear log-worthy reason why -- never a crash, never
fabricated numbers.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NVMLSample:
    timestamp: float
    power_watts: Optional[float]
    sm_clock_mhz: Optional[int]
    memory_clock_mhz: Optional[int]
    temperature_c: Optional[int]
    memory_used_mb: Optional[float]
    memory_total_mb: Optional[float]
    throttle_reasons: List[str]


class NVMLSampler:
    """Background sampling thread. `start()`/`stop()` bracket a run;
    `samples` holds everything collected in between. If pynvml can't be
    initialized (no driver, not installed), `unavailable_reason` explains
    why and every call to `start()` is a no-op that produces no samples --
    calling code doesn't need an if-GPU-available branch of its own."""

    def __init__(self, interval_s: float = 1.0, device_index: int = 0):
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples: List[NVMLSample] = []
        self.unavailable_reason: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._handle = None
        self._pynvml = None

    def _try_init(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            return True
        except Exception as e:  # noqa: BLE001 - any failure here means "no GPU telemetry", not a crash
            self.unavailable_reason = f"{type(e).__name__}: {e}"
            return False

    def _sample_once(self) -> NVMLSample:
        pynvml = self._pynvml
        handle = self._handle

        def _safe(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - a single failed metric shouldn't drop the whole sample
                return None

        power = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        sm_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        mem_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        temp = _safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        mem_info = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        throttle_bitmask = _safe(lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))

        reasons = []
        if throttle_bitmask:
            for name in dir(pynvml):
                if name.startswith("nvmlClocksThrottleReason"):
                    bit = getattr(pynvml, name)
                    if throttle_bitmask & bit:
                        reasons.append(name.replace("nvmlClocksThrottleReason", ""))

        return NVMLSample(
            timestamp=time.time(),
            power_watts=power,
            sm_clock_mhz=sm_clock,
            memory_clock_mhz=mem_clock,
            temperature_c=temp,
            memory_used_mb=(mem_info.used / (1024 * 1024)) if mem_info else None,
            memory_total_mb=(mem_info.total / (1024 * 1024)) if mem_info else None,
            throttle_reasons=reasons,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.samples.append(self._sample_once())

    def start(self) -> None:
        self.samples = []
        self._stop_event.clear()
        if not self._try_init():
            return  # unavailable_reason is set; start() is a documented no-op
        self._thread = threading.Thread(target=self._run, name="nvml-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5.0)
            self._thread = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
