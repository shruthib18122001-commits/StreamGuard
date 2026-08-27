"""Real NVDEC hardware decode via VideoProcessingFramework's `PyNvCodec`
binding.

UNVERIFIED, and less certain than any other GPU-only file in this repo:
NVIDIA's NVDEC Python bindings (VPF's `PyNvCodec` vs. the newer standalone
`PyNvVideoCodec` package) have shifted more across versions than TensorRT's
API has. This is written against VPF's `PyNvDecoder` / `DecodeSingleSurface`
/ `PySurfaceDownloader` surface, which has the longest track record, but
treat the exact method names here as the first thing to re-verify against
whatever's actually installed on Colab -- more so than tensorrt_backend.py
or cuda_timer.py, where the underlying APIs are considerably more stable.

Kept alongside CPUDecoder behind the same Decoder interface specifically so
the NVDEC-vs-CPU-decode comparison (CLAUDE.md's experiments list) has a
real baseline to measure against, and so a wrong detail here is a
contained, swappable implementation -- not a rewrite of anything else.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from streamguard.decode.base import Decoder


class NVDECDecoder(Decoder):
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._decoder = None
        self._downloader = None

    def open(self, source: str) -> None:
        import PyNvCodec as nvc

        self._decoder = nvc.PyNvDecoder(source, self.gpu_id)
        # Downloads a decoded surface (device-resident, NV12) to a host
        # numpy array. Frame stays device-resident from NVDEC through
        # preprocessing in the eventual full GPU path (nvdec_gpu_preprocess
        # below); this CPU-decode-shaped Decoder interface downloads here
        # only so NVDECDecoder is a drop-in swap for CPUDecoder without
        # changing every caller -- the throughput comparison this exists
        # for is decode-engine vs. decode-engine, not "GPU vs CPU
        # end-to-end," so the extra download here doesn't undermine it.
        self._downloader = nvc.PySurfaceDownloader(
            self._decoder.Width(), self._decoder.Height(), nvc.PixelFormat.NV12, self.gpu_id
        )

    def read_frame(self) -> Optional[np.ndarray]:
        if self._decoder is None:
            raise RuntimeError("NVDECDecoder.read_frame() called before open()")

        surface = self._decoder.DecodeSingleSurface()
        if surface is None or surface.Empty():
            return None

        width, height = self._decoder.Width(), self._decoder.Height()
        nv12_frame = np.ndarray(shape=(height * 3 // 2, width), dtype=np.uint8)
        if not self._downloader.DownloadSingleSurface(surface, nv12_frame):
            raise RuntimeError("failed to download decoded surface from GPU")

        import cv2

        return cv2.cvtColor(nv12_frame, cv2.COLOR_YUV2BGR_NV12)

    def close(self) -> None:
        self._decoder = None
        self._downloader = None
