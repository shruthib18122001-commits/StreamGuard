import time

import numpy as np
import pytest

from streamguard.config import ModelTier, Precision
from streamguard.inference.mock import MockInferenceBackend


def make_batch(n: int, size: int = 64) -> np.ndarray:
    return np.zeros((n, size, size, 3), dtype=np.uint8)


def test_infer_before_load_raises():
    backend = MockInferenceBackend()
    with pytest.raises(RuntimeError):
        backend.infer(make_batch(1))


def test_unconfigured_tier_precision_raises():
    backend = MockInferenceBackend(latency_table={})
    with pytest.raises(ValueError):
        backend.load(ModelTier.M, Precision.FP16, 640)


def test_infer_returns_one_detection_list_per_frame():
    backend = MockInferenceBackend(rng_seed=0)
    backend.load(ModelTier.NANO, Precision.INT8, 416)
    batch = make_batch(4, size=416)
    results = backend.infer(batch)
    assert len(results) == 4
    for dets in results:
        assert isinstance(dets, list)
        for d in dets:
            assert 0 <= d.class_id < 80
            assert 0.3 <= d.confidence <= 0.99


def test_infer_latency_roughly_matches_configured_tier():
    # NOTE: this measures wall-clock time to sanity-check the *mock's* sleep,
    # not GPU work — there's no GPU here. It is not a stand-in for the
    # CUDA-event timing real inference stages must use later.
    backend = MockInferenceBackend(rng_seed=1)
    backend.load(ModelTier.NANO, Precision.INT8, 416)  # ~3ms configured
    start = time.perf_counter()
    backend.infer(make_batch(1, size=416))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms >= 2.0  # allow scheduler slack below the 3ms target


def test_higher_tier_is_slower_than_lower_tier():
    backend = MockInferenceBackend(rng_seed=2)

    backend.load(ModelTier.NANO, Precision.INT8, 416)
    start = time.perf_counter()
    backend.infer(make_batch(1, size=416))
    nano_ms = (time.perf_counter() - start) * 1000.0

    backend.load(ModelTier.M, Precision.FP32, 640)
    start = time.perf_counter()
    backend.infer(make_batch(1, size=640))
    m_ms = (time.perf_counter() - start) * 1000.0

    assert m_ms > nano_ms


def test_current_config_reflects_last_load():
    backend = MockInferenceBackend()
    backend.load(ModelTier.S, Precision.INT8, 640)
    cfg = backend.current_config
    assert cfg["tier"] == ModelTier.S
    assert cfg["precision"] == Precision.INT8
    assert cfg["input_size"] == 640
