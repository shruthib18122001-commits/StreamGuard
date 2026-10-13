#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX export of a YOLOX model.

UNVERIFIED on this dev machine (no CUDA/TensorRT here) -- see the stage-4
validation checklist in the README before trusting this against real data.
Produces `{tier}_{precision}_{input_size}.engine` in --output-dir, matching
the filename streamguard.inference.tensorrt_backend.TensorRTInferenceBackend
expects.

Prerequisite this script does NOT handle: exporting a YOLOX checkpoint to
ONNX. That needs YOLOX's own tools/export_onnx.py (it requires YOLOX's
model-definition code to retrace the graph) -- pulling that dependency into
this repo just to re-derive ONNX export felt like the wrong call, so this
script starts from an already-exported .onnx file.

INT8 requires --calib-dir: a folder of representative frames (.jpg/.png),
never committed to this repo (see .gitignore) -- a few hundred images
spanning the real deployment distribution is the usual guidance.

Usage:
    python scripts/build_engine.py --onnx yolox_nano.onnx --output-dir engines \
        --tier nano --precision int8 --input-size 416 --calib-dir calib_images/
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import tensorrt as trt

from streamguard.config import ModelTier, Precision
from streamguard.preprocess.cpu import CPUPreprocessor


class _Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """Feeds batches of representative preprocessed frames to TensorRT's
    INT8 calibration pass, and caches the resulting scale factors to disk
    so re-running the build doesn't re-calibrate from scratch."""

    def __init__(self, calib_dir: str, input_size: int, batch_size: int, cache_path: str):
        super().__init__()
        import cupy as cp

        self._cache_path = cache_path
        self._batch_size = batch_size
        self._preprocessor = CPUPreprocessor()
        self._input_size = input_size

        image_paths = sorted(
            glob.glob(os.path.join(calib_dir, "*.jpg")) + glob.glob(os.path.join(calib_dir, "*.png"))
        )
        if not image_paths:
            raise ValueError(f"no .jpg/.png calibration images found in {calib_dir!r}")
        self._image_paths = image_paths
        self._batch_index = 0
        self._device_input = cp.empty((batch_size, 3, input_size, input_size), dtype=cp.float32)  # NCHW

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        import cv2

        start = self._batch_index * self._batch_size
        if start >= len(self._image_paths):
            return None  # signals calibration is complete

        batch_paths = self._image_paths[start : start + self._batch_size]
        frames = []
        for p in batch_paths:
            frame = cv2.imread(p)
            if frame is None:
                raise ValueError(f"could not read calibration image {p!r}")
            processed, _ = self._preprocessor.process(frame, self._input_size)
            frames.append(processed.transpose(2, 0, 1).astype(np.float32))  # HWC -> CHW, raw 0-255 like the backend
        while len(frames) < self._batch_size:
            frames.append(frames[-1])  # pad the final partial batch by repeating the last frame

        self._device_input.set(np.stack(frames, axis=0))
        self._batch_index += 1
        return [int(self._device_input.data.ptr)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        with open(self._cache_path, "wb") as f:
            f.write(cache)


def build_engine(
    onnx_path: str,
    output_path: str,
    precision: Precision,
    input_size: int,
    calib_dir: Optional[str] = None,
    calib_cache_path: Optional[str] = None,
    calib_batch_size: int = 8,
    max_batch_size: int = 16,
) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"failed to parse ONNX model {onnx_path!r}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    input_tensor = network.get_input(0)
    in_shape = tuple(input_tensor.shape)
    if len(in_shape) != 4 or in_shape[1] != 3:
        raise RuntimeError(f"ONNX input {input_tensor.name!r} has shape {in_shape}, expected NCHW (N, 3, H, W)")
    if in_shape[0] != -1:
        raise RuntimeError(
            f"ONNX input {input_tensor.name!r} has static batch {in_shape[0]}; the dynamic batcher needs a "
            "dynamic batch axis. Re-export with YOLOX tools/export_onnx.py --dynamic."
        )
    profile = builder.create_optimization_profile()
    # NCHW: that is what YOLOX's ONNX export produces. The ONNX must have been
    # exported with --dynamic (batch axis dynamic); a static batch-1 export
    # cannot take a min!=max profile.
    min_shape = (1, 3, input_size, input_size)
    opt_shape = (max(1, max_batch_size // 2), 3, input_size, input_size)
    max_shape = (max_batch_size, 3, input_size, input_size)
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if precision == Precision.FP16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this GPU has no fast FP16 support -- check the target device")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == Precision.INT8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("this GPU has no fast INT8 support -- check the target device")
        if calib_dir is None:
            raise ValueError("precision=int8 requires --calib-dir (a folder of representative frames)")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow FP16 fallback for layers INT8 handles poorly
        config.int8_calibrator = _Int8Calibrator(
            calib_dir, input_size, calib_batch_size,
            calib_cache_path or output_path + ".calib_cache",
        )
    # FP32 needs no flags -- it's TensorRT's default working precision.

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"engine build failed for {onnx_path!r} at precision={precision.value}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)
    print(f"wrote {output_path} ({len(serialized_engine) / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, help="path to the exported YOLOX ONNX model")
    parser.add_argument("--output-dir", required=True, help="directory to write the .engine file into")
    parser.add_argument("--tier", required=True, choices=[t.value for t in ModelTier])
    parser.add_argument("--precision", required=True, choices=[p.value for p in Precision])
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--calib-dir", default=None, help="required when --precision int8")
    parser.add_argument("--calib-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=16)
    args = parser.parse_args()

    tier = ModelTier(args.tier)
    precision = Precision(args.precision)
    output_path = os.path.join(args.output_dir, f"{tier.value}_{precision.value}_{args.input_size}.engine")

    build_engine(
        onnx_path=args.onnx,
        output_path=output_path,
        precision=precision,
        input_size=args.input_size,
        calib_dir=args.calib_dir,
        calib_batch_size=args.calib_batch_size,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
