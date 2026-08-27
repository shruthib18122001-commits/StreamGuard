"""Real TensorRT execution backend, implementing the same InferenceBackend
interface the mock backend does (stage 1) -- the whole pipeline was built
against that interface specifically so this swap-in requires no pipeline
changes.

UNVERIFIED: there's no NVIDIA driver, CUDA, or TensorRT on this dev
machine, so none of this has actually been run. It's written against the
modern (TensorRT 8.5+) execute_async_v3 / tensor-address API, based on
documented behavior, not a working test. Before trusting it against real
data, run the validation checklist in the README's stage-4 section on
Colab -- start there, not with a full experiment sweep.

Threading contract (CLAUDE.md): the compiled Engine is safe to share
read-only across threads; the ExecutionContext this class owns is not.
Create one TensorRTInferenceBackend per worker thread and never share one
instance's context across threads. The pipeline today only ever runs one
inference thread, so this is trivially satisfied right now -- worth
stating explicitly anyway, since it stops being trivial the moment that
changes.

Uses CuPy (not pycuda) for all device memory: it's already the intended
tool for GPU preprocessing, so this avoids a second GPU-memory library.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from streamguard.config import ModelTier, Precision
from streamguard.inference.base import Detection, InferenceBackend
from streamguard.inference.yolox_decode import decode_yolox_output

_trt_logger = None  # created lazily by _get_logger(), which needs `tensorrt` importable


def _get_logger():
    global _trt_logger
    if _trt_logger is None:
        import tensorrt as trt

        _trt_logger = trt.Logger(trt.Logger.WARNING)
    return _trt_logger


class TensorRTInferenceBackend(InferenceBackend):
    def __init__(self, engine_dir: str, num_classes: int = 80, conf_threshold: float = 0.3):
        """engine_dir: directory containing engines named
        `{tier}_{precision}_{input_size}.engine`, produced by
        scripts/build_engine.py -- one per (tier, precision, size)
        combination the degradation ladder can select."""
        self._engine_dir = engine_dir
        self._num_classes = num_classes
        self._conf_threshold = conf_threshold

        self._engine = None
        self._context = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._input_size: Optional[int] = None
        self._stream = None

        # Allocated on first infer() for a given batch size; reused across
        # calls when the shape doesn't change. See _ensure_buffers().
        self._d_input = None
        self._d_output = None
        self._h_input_pinned = None
        self._h_output_pinned = None
        self._output_shape: Optional[Tuple[int, ...]] = None

    def _engine_path(self, tier: ModelTier, precision: Precision, input_size: int) -> str:
        return os.path.join(self._engine_dir, f"{tier.value}_{precision.value}_{input_size}.engine")

    def load(self, tier: ModelTier, precision: Precision, input_size: int) -> None:
        path = self._engine_path(tier, precision, input_size)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no engine at {path!r} -- build it first with scripts/build_engine.py "
                f"(--tier {tier.value} --precision {precision.value} --input-size {input_size})"
            )

        import cupy as cp
        import tensorrt as trt

        runtime = trt.Runtime(_get_logger())
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine at {path!r}")

        self._engine = engine
        self._context = engine.create_execution_context()
        self._input_size = input_size
        self._stream = cp.cuda.Stream(non_blocking=True)

        io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inputs = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"expected exactly one input and one output tensor, got inputs={inputs} outputs={outputs} "
                "-- this backend assumes a single-input/single-output YOLOX-style export"
            )
        self._input_name, self._output_name = inputs[0], outputs[0]

        # Force reallocation of device/pinned buffers for the new engine on
        # the next infer() call, even if a previous engine used the same
        # batch size -- shapes/dtypes may differ between tiers.
        self._d_input = None
        self._d_output = None

    def _ensure_buffers(self, batch_size: int) -> None:
        import cupy as cp

        input_shape = (batch_size, self._input_size, self._input_size, 3)
        if self._d_input is not None and self._d_input.shape == input_shape:
            return

        self._context.set_input_shape(self._input_name, input_shape)
        output_shape = tuple(self._context.get_tensor_shape(self._output_name))
        if any(d < 0 for d in output_shape):
            raise RuntimeError(
                f"TensorRT couldn't resolve a concrete output shape for input {input_shape} "
                f"(got {output_shape}) -- does this engine's optimization profile cover this batch size?"
            )

        self._d_input = cp.empty(input_shape, dtype=cp.float32)
        self._d_output = cp.empty(output_shape, dtype=cp.float32)
        # Pinned (page-locked) host memory, per CLAUDE.md: without it,
        # cudaMemcpyAsync silently falls back to a synchronous copy and the
        # decode/preprocess/inference overlap this architecture depends on
        # disappears.
        self._h_input_pinned = cp.cuda.alloc_pinned_memory(self._d_input.nbytes)
        self._h_output_pinned = cp.cuda.alloc_pinned_memory(self._d_output.nbytes)
        self._output_shape = output_shape

    def infer(self, batch: np.ndarray) -> List[List[Detection]]:
        if self._context is None:
            raise RuntimeError("TensorRTInferenceBackend.infer() called before load()")

        batch_size = batch.shape[0]
        self._ensure_buffers(batch_size)

        host_in = np.frombuffer(self._h_input_pinned, dtype=np.float32).reshape(self._d_input.shape)
        np.copyto(host_in, batch.astype(np.float32))

        self._d_input.set(host_in, stream=self._stream)  # pinned H2D copy, async on self._stream
        self._context.set_tensor_address(self._input_name, int(self._d_input.data.ptr))
        self._context.set_tensor_address(self._output_name, int(self._d_output.data.ptr))
        self._context.execute_async_v3(self._stream.ptr)

        host_out = np.frombuffer(self._h_output_pinned, dtype=np.float32).reshape(self._output_shape)
        self._d_output.get(out=host_out, stream=self._stream)
        self._stream.synchronize()  # block until the D2H copy above has actually landed

        return decode_yolox_output(host_out, self._input_size, self._conf_threshold)
