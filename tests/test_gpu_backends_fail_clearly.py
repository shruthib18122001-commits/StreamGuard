"""These two backends need real CUDA/TensorRT/CuPy, none of which exist on
this dev machine. The point of these tests isn't to validate GPU
correctness -- that's impossible here -- it's to lock in a narrower promise:
importing streamguard.inference / streamguard.metrics must not require
those libraries (imports are deferred into methods), and calling into a GPU
backend without its prerequisites must fail with a clear, specific error
instead of an obscure one three layers down."""
import pytest


def test_tensorrt_backend_module_imports_without_gpu_libs():
    from streamguard.inference.tensorrt_backend import TensorRTInferenceBackend

    assert TensorRTInferenceBackend is not None


def test_tensorrt_backend_load_fails_clearly(tmp_path):
    from streamguard.config import ModelTier, Precision
    from streamguard.inference.tensorrt_backend import TensorRTInferenceBackend

    backend = TensorRTInferenceBackend(engine_dir=str(tmp_path))
    # Either tensorrt isn't installed here (ModuleNotFoundError) or, on a
    # machine that does have it, the engine file simply doesn't exist yet
    # (FileNotFoundError) -- both are acceptable, "silently returns garbage"
    # is not.
    with pytest.raises((FileNotFoundError, ModuleNotFoundError, ImportError)):
        backend.load(ModelTier.NANO, Precision.INT8, 416)


def test_tensorrt_backend_infer_before_load_raises(tmp_path):
    from streamguard.inference.tensorrt_backend import TensorRTInferenceBackend

    backend = TensorRTInferenceBackend(engine_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        backend.infer(None)


def test_cuda_event_timer_module_imports_without_gpu():
    from streamguard.metrics.cuda_timer import CudaEventTimer

    assert CudaEventTimer is not None


def test_cuda_event_timer_instantiation_fails_clearly_without_cupy():
    from streamguard.metrics.cuda_timer import CudaEventTimer

    with pytest.raises((ModuleNotFoundError, ImportError)):
        CudaEventTimer()


def test_nvdec_decoder_module_imports_without_gpu_libs():
    from streamguard.decode.nvdec import NVDECDecoder

    assert NVDECDecoder is not None


def test_nvdec_decoder_open_fails_clearly_without_pynvcodec():
    from streamguard.decode.nvdec import NVDECDecoder

    decoder = NVDECDecoder()
    with pytest.raises((ModuleNotFoundError, ImportError)):
        decoder.open("does-not-matter.mp4")


def test_gpu_preprocessor_module_imports_without_cupy():
    from streamguard.preprocess.gpu import GPUPreprocessor

    assert GPUPreprocessor is not None


def test_gpu_preprocessor_process_fails_clearly_without_cupy():
    from streamguard.preprocess.gpu import GPUPreprocessor

    with pytest.raises((ModuleNotFoundError, ImportError)):
        GPUPreprocessor().process(None, 640)
