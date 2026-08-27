"""GPU letterbox-resize preprocessor via CuPy -- keeps the frame
device-resident from decode through inference, per CLAUDE.md's "frame
stays device-resident from decode to output."

UNVERIFIED: no CUDA device here to run this against. Same letterbox math as
CPUPreprocessor (aspect-preserving resize, gray-114 pad), just executed as
CuPy device-array ops instead of cv2 host-array ops, so results should
match CPUPreprocessor exactly given the same input -- that equivalence
(same output for a real frame, GPU vs CPU) is one of the first things
worth checking once this runs somewhere with a GPU.

Takes and returns CuPy arrays, not numpy: the caller is responsible for
getting the frame onto the device in the first place (e.g. from
NVDECDecoder's output) and for whatever happens after (batching, TensorRT
input) staying device-resident too -- this class doesn't do any host
transfer itself.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from streamguard.preprocess.base import Preprocessor, PreprocessMeta

if TYPE_CHECKING:
    import cupy as cp

LETTERBOX_PAD_VALUE = 114


class GPUPreprocessor(Preprocessor):
    def process(self, frame: "cp.ndarray", target_size: int) -> Tuple["cp.ndarray", PreprocessMeta]:
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage

        orig_h, orig_w = frame.shape[:2]
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))

        # cupyx's zoom is the device-side equivalent of cv2.resize here;
        # order=1 matches cv2.INTER_LINEAR (bilinear).
        zoom_factors = (new_h / orig_h, new_w / orig_w, 1.0)
        resized = cupy_ndimage.zoom(frame, zoom_factors, order=1)
        resized = resized[:new_h, :new_w, :]  # zoom's rounding can overshoot by a pixel; clip to exact target

        canvas = cp.full((target_size, target_size, 3), LETTERBOX_PAD_VALUE, dtype=cp.uint8)
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
