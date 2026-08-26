"""Preprocessor interface.

CPU/numpy implementation now; the GPU/CuPy path (frame stays device-resident
from decode through output, per CLAUDE.md) arrives once there's a real device
to keep it resident on. Same interface either way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PreprocessMeta:
    """Enough information to map a detection box in letterboxed model-input
    coordinates back to the original frame's coordinates."""

    scale: float
    pad_x: float
    pad_y: float
    orig_width: int
    orig_height: int


class Preprocessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray, target_size: int) -> Tuple[np.ndarray, PreprocessMeta]:
        """Letterbox-resize `frame` to (target_size, target_size, 3) and
        return it alongside the metadata needed to invert the transform."""
