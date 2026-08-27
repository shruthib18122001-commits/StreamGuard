import numpy as np
import pytest

from streamguard.inference.yolox_decode import YOLOX_STRIDES, _grids_and_strides, decode_yolox_output


def total_anchors(input_size: int) -> int:
    return sum((input_size // s) ** 2 for s in YOLOX_STRIDES)


def test_grids_and_strides_anchor_count_640():
    grids, strides = _grids_and_strides(640)
    n = total_anchors(640)
    assert grids.shape == (n, 2)
    assert strides.shape == (n, 1)


def test_grids_and_strides_anchor_count_416():
    grids, strides = _grids_and_strides(416)
    assert grids.shape == (total_anchors(416), 2)


def test_first_anchor_is_grid_zero_stride_8():
    grids, strides = _grids_and_strides(640)
    assert tuple(grids[0]) == (0.0, 0.0)
    assert strides[0, 0] == 8.0


def test_decode_single_known_anchor():
    input_size = 640
    num_classes = 2
    raw = np.zeros((1, total_anchors(input_size), 5 + num_classes), dtype=np.float64)

    # Anchor 0 is grid (0, 0) at stride 8 (see _grids_and_strides ordering).
    raw[0, 0, 0] = 0.5  # reg_x -> cx = (0.5 + 0) * 8 = 4.0
    raw[0, 0, 1] = 0.5  # reg_y -> cy = 4.0
    raw[0, 0, 2] = 0.0  # reg_w -> exp(0) * 8 = 8.0
    raw[0, 0, 3] = 0.0  # reg_h -> 8.0
    raw[0, 0, 4] = 0.9  # obj
    raw[0, 0, 5] = 0.05  # cls 0
    raw[0, 0, 6] = 0.9  # cls 1

    result = decode_yolox_output(raw, input_size=input_size, conf_threshold=0.3)
    assert len(result) == 1
    dets = result[0]
    assert len(dets) == 1
    d = dets[0]
    assert d.x1 == pytest.approx(0.0)
    assert d.y1 == pytest.approx(0.0)
    assert d.x2 == pytest.approx(8.0)
    assert d.y2 == pytest.approx(8.0)
    assert d.confidence == pytest.approx(0.9 * 0.9)
    assert d.class_id == 1


def test_decode_filters_below_confidence_threshold():
    input_size = 640
    raw = np.zeros((1, total_anchors(input_size), 7), dtype=np.float64)
    raw[0, 0, 4] = 0.1  # low objectness everywhere -> nothing survives
    result = decode_yolox_output(raw, input_size=input_size, conf_threshold=0.3)
    assert result == [[]]


def test_decode_handles_batch_of_multiple_frames():
    input_size = 416
    raw = np.zeros((2, total_anchors(input_size), 8), dtype=np.float64)
    raw[0, 0, 4] = 0.9
    raw[0, 0, 5] = 0.9  # frame 0: one detection, class 0
    raw[1, 0, 4] = 0.0  # frame 1: nothing

    result = decode_yolox_output(raw, input_size=input_size, conf_threshold=0.3)
    assert len(result) == 2
    assert len(result[0]) == 1
    assert result[0][0].class_id == 0
    assert len(result[1]) == 0


def test_decode_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        decode_yolox_output(np.zeros((10, 7)), input_size=640)


def test_decode_rejects_anchor_count_mismatch():
    raw = np.zeros((1, 100, 7), dtype=np.float64)  # wrong anchor count for input_size=640
    with pytest.raises(ValueError):
        decode_yolox_output(raw, input_size=640)


def test_decode_rejects_too_few_channels():
    raw = np.zeros((1, total_anchors(640), 5), dtype=np.float64)  # implies 0 classes
    with pytest.raises(ValueError):
        decode_yolox_output(raw, input_size=640)
