"""CPU letterbox-resize preprocessor.

Standard YOLOX-style preprocessing: resize preserving aspect ratio, pad the
remainder with gray (114) rather than black — a YOLOX/COCO-training
convention, not an arbitrary choice, since the model was trained expecting
that pad value. cv2.resize is a vectorized C call, not a per-pixel Python
loop, so this stays within the "no heavy per-frame Python work" rule even
running on CPU.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from streamguard.preprocess.base import Preprocessor, PreprocessMeta

LETTERBOX_PAD_VALUE = 114


class CPUPreprocessor(Preprocessor):
    def process(self, frame: np.ndarray, target_size: int) -> Tuple[np.ndarray, PreprocessMeta]:
        orig_h, orig_w = frame.shape[:2]
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_size, target_size, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        meta = PreprocessMeta(
            scale=scale,
            pad_x=float(pad_x),
            pad_y=float(pad_y),
            orig_width=orig_w,
            orig_height=orig_h,
        )
        return canvas, meta
