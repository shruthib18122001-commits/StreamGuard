"""Device/driver/TensorRT version snapshot.

"Every result row must carry device, driver version, and TensorRT version"
(CLAUDE.md) — this is that snapshot. It's a best-effort probe: on the Mac dev
machine there's no NVIDIA driver and no pynvml/tensorrt installed, so every
GPU-specific field comes back None rather than a guess. Continuous NVML
sampling (power/clocks/temp/throttle) is separate, real-time telemetry that
belongs alongside the actual GPU execution loop in a later stage — there's
nothing to sample yet, so it isn't stubbed here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    device: str
    gpu_name: Optional[str] = None
    driver_version: Optional[str] = None
    tensorrt_version: Optional[str] = None
    cuda_runtime_version: Optional[str] = None  # what CuPy/the app links against
    cuda_driver_api_version: Optional[str] = None  # max CUDA the installed driver supports
    captured_at: float = 0.0


def _probe_gpu_name_and_driver():
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
            return name, driver
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None, None


def _probe_tensorrt_version():
    try:
        import tensorrt as trt

        return trt.__version__
    except Exception:
        return None


def _probe_cuda_versions():
    """CuPy reports versions as ints, e.g. 12020 -> 12.2."""
    try:
        import cupy as cp

        def fmt(v: int) -> str:
            return f"{v // 1000}.{(v % 1000) // 10}"

        return fmt(cp.cuda.runtime.runtimeGetVersion()), fmt(cp.cuda.runtime.driverGetVersion())
    except Exception:
        return None, None


def capture_device_info(device: str) -> DeviceInfo:
    gpu_name, driver_version = _probe_gpu_name_and_driver()
    tensorrt_version = _probe_tensorrt_version()
    cuda_runtime, cuda_driver_api = _probe_cuda_versions()
    return DeviceInfo(
        device=device,
        gpu_name=gpu_name,
        driver_version=driver_version,
        tensorrt_version=tensorrt_version,
        cuda_runtime_version=cuda_runtime,
        cuda_driver_api_version=cuda_driver_api,
        captured_at=time.time(),
    )
