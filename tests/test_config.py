from pathlib import Path

import pytest
from pydantic import ValidationError

from streamguard.config import (
    ModelTier,
    Precision,
    SLAConfig,
    StreamConfig,
    StreamGuardConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_default_yaml():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg.device == "cuda:0"
    assert cfg.streams.count == 4
    assert cfg.model.tier == ModelTier.M
    assert cfg.model.precision == Precision.FP16
    assert cfg.sla.deadline_ms == 33.0
    assert len(cfg.sla.ladder) == 6


def test_defaults_construct_without_yaml():
    cfg = StreamGuardConfig()
    assert cfg.streams.count == 1
    assert cfg.sla.ladder[0].level == 0
    assert cfg.sla.ladder[-1].level == 5


def test_hysteresis_validation_rejects_inverted_thresholds():
    with pytest.raises(ValidationError):
        SLAConfig(degrade_threshold_ms=20.0, recover_threshold_ms=20.0)
    with pytest.raises(ValidationError):
        SLAConfig(degrade_threshold_ms=10.0, recover_threshold_ms=20.0)


def test_stream_count_must_be_positive():
    with pytest.raises(ValidationError):
        StreamConfig(count=0)


def test_input_size_must_be_positive():
    from streamguard.config import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(input_size=0)


def test_ladder_levels_must_be_sorted_and_unique():
    from streamguard.config import DegradationTier

    bad_ladder = [
        DegradationTier(level=0, model_tier=ModelTier.M, precision=Precision.FP16, input_size=640),
        DegradationTier(level=0, model_tier=ModelTier.S, precision=Precision.FP16, input_size=640),
    ]
    with pytest.raises(ValidationError):
        SLAConfig(ladder=bad_ladder)
