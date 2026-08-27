"""Decode-engine throughput comparison: CPU decode vs NVDEC (CLAUDE.md's
"NVDEC vs CPU decode: throughput and CPU utilization"). Only the CPU side
is runnable on this dev machine; the harness works with any two Decoder
factories, so swapping in NVDECDecoder once it's verified on Colab is a
one-line change here, not a rewrite.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from streamguard.decode.base import Decoder


@dataclass
class DecodeThroughputResult:
    decoder_name: str
    frames_decoded: int
    wall_seconds: float
    fps: float
    mean_cpu_percent: Optional[float]


def measure_decode_throughput(
    decoder_factory: Callable[[], Decoder],
    source: str,
    decoder_name: str,
    max_frames: Optional[int] = None,
) -> DecodeThroughputResult:
    import psutil

    process = psutil.Process()
    process.cpu_percent()  # primes psutil's internal interval timer; first call always returns 0.0

    decoder = decoder_factory()
    decoder.open(source)
    frames = 0
    start = time.perf_counter()
    try:
        while max_frames is None or frames < max_frames:
            frame = decoder.read_frame()
            if frame is None:
                break
            frames += 1
    finally:
        decoder.close()
    wall_s = time.perf_counter() - start
    mean_cpu_percent = process.cpu_percent()  # average utilization over the interval since the call above

    return DecodeThroughputResult(
        decoder_name=decoder_name,
        frames_decoded=frames,
        wall_seconds=wall_s,
        fps=(frames / wall_s) if wall_s > 0 else 0.0,
        mean_cpu_percent=mean_cpu_percent,
    )


def compare_decoders(
    decoders: List[Tuple[str, Callable[[], Decoder]]],
    source: str,
    max_frames: Optional[int] = None,
) -> List[DecodeThroughputResult]:
    return [measure_decode_throughput(factory, source, name, max_frames) for name, factory in decoders]
