"""Decoder interface.

Real CPU decode now (CPUDecoder, below); NVDEC is deliberately not implemented
yet. There's no NVIDIA driver on this dev machine to test a binding against,
and unlike TensorRT (locked-in, unavoidable, arriving stage 4 regardless),
GPU-vs-CPU decode is an experiment to run later, not a stage-2 requirement.
NVDECDecoder lands once we're actually on Colab and can verify a library
against real hardware instead of guessing blind.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Decoder(ABC):
    @abstractmethod
    def open(self, source: str) -> None:
        """Open a video source (file path or stream URI)."""

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Return the next frame as an (H, W, 3) BGR uint8 array, or None at
        end of stream."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the decoder."""
