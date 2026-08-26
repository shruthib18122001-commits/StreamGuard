import pytest

from streamguard.inference.base import Detection
from streamguard.postprocess import Postprocessor
from streamguard.preprocess.base import PreprocessMeta


def identity_meta(width=100, height=100) -> PreprocessMeta:
    return PreprocessMeta(scale=1.0, pad_x=0.0, pad_y=0.0, orig_width=width, orig_height=height)


def test_confidence_filter_removes_low_score_detections():
    dets = [
        Detection(x1=10, y1=10, x2=20, y2=20, confidence=0.1, class_id=0),
        Detection(x1=30, y1=30, x2=40, y2=40, confidence=0.9, class_id=0),
    ]
    result = Postprocessor(conf_threshold=0.3).process(dets, identity_meta())
    assert len(result) == 1
    assert result[0].confidence == pytest.approx(0.9)


def test_nms_suppresses_heavily_overlapping_lower_confidence_box():
    dets = [
        Detection(x1=10, y1=10, x2=50, y2=50, confidence=0.9, class_id=0),
        Detection(x1=12, y1=12, x2=52, y2=52, confidence=0.8, class_id=0),  # IoU ~0.82
    ]
    result = Postprocessor(conf_threshold=0.3, iou_threshold=0.45).process(dets, identity_meta())
    assert len(result) == 1
    assert result[0].confidence == pytest.approx(0.9)


def test_nms_keeps_non_overlapping_boxes():
    dets = [
        Detection(x1=10, y1=10, x2=20, y2=20, confidence=0.9, class_id=0),
        Detection(x1=60, y1=60, x2=70, y2=70, confidence=0.8, class_id=0),
    ]
    result = Postprocessor(conf_threshold=0.3).process(dets, identity_meta())
    assert len(result) == 2


def test_coordinate_inversion_round_trip():
    # Wide-frame letterbox scenario: 320x240 orig -> 416 target -> scale 1.3, pad_y 52
    scale, pad_x, pad_y = 1.3, 0.0, 52.0
    meta = PreprocessMeta(scale=scale, pad_x=pad_x, pad_y=pad_y, orig_width=320, orig_height=240)

    orig_box = (50.0, 30.0, 150.0, 120.0)
    letterboxed = tuple(
        c * scale + (pad_x if i % 2 == 0 else pad_y) for i, c in enumerate(orig_box)
    )
    dets = [Detection(x1=letterboxed[0], y1=letterboxed[1], x2=letterboxed[2], y2=letterboxed[3],
                       confidence=0.9, class_id=5)]

    result = Postprocessor(conf_threshold=0.3).process(dets, meta)
    assert len(result) == 1
    r = result[0]
    assert r.x1 == pytest.approx(orig_box[0], abs=1e-6)
    assert r.y1 == pytest.approx(orig_box[1], abs=1e-6)
    assert r.x2 == pytest.approx(orig_box[2], abs=1e-6)
    assert r.y2 == pytest.approx(orig_box[3], abs=1e-6)
    assert r.class_id == 5


def test_empty_detections_returns_empty():
    assert Postprocessor().process([], identity_meta()) == []


def test_boxes_clipped_to_original_frame_bounds():
    dets = [Detection(x1=-10, y1=-10, x2=150, y2=150, confidence=0.9, class_id=0)]
    result = Postprocessor(conf_threshold=0.3).process(dets, identity_meta())
    assert result[0].x1 == 0.0
    assert result[0].y1 == 0.0
    assert result[0].x2 == 100.0
    assert result[0].y2 == 100.0
