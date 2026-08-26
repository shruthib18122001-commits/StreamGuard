"""CPU decode path via OpenCV — kept even after NVDEC exists, so the
NVDEC-vs-CPU throughput/CPU-utilization comparison (CLAUDE.md experiments)
has a real baseline to measure against, not just a synthetic one."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from streamguard.decode.base import Decoder


class CPUDecoder(Decoder):
    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self, source: str) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video source: {source!r}")
        self._cap = cap

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("CPUDecoder.read_frame() called before open()")
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
