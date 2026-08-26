import numpy as np
import pytest

from streamguard.preprocess.cpu import CPUPreprocessor, LETTERBOX_PAD_VALUE


def test_output_shape_is_square_target_size():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    canvas, meta = CPUPreprocessor().process(frame, target_size=416)
    assert canvas.shape == (416, 416, 3)
    assert canvas.dtype == np.uint8


def test_wide_frame_pads_top_and_bottom():
    # 320x240 (wider than tall) resized into a 416x416 square must pad
    # vertically, not horizontally.
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    canvas, meta = CPUPreprocessor().process(frame, target_size=416)

    assert meta.pad_y > 0
    assert meta.pad_x == 0
    assert meta.orig_width == 320
    assert meta.orig_height == 240
    assert meta.scale == pytest.approx(416 / 320)

    # padded strip should still be the letterbox gray value
    assert np.all(canvas[0, :, :] == LETTERBOX_PAD_VALUE)


def test_tall_frame_pads_left_and_right():
    frame = np.zeros((320, 240, 3), dtype=np.uint8)
    canvas, meta = CPUPreprocessor().process(frame, target_size=416)

    assert meta.pad_x > 0
    assert meta.pad_y == 0
    assert meta.scale == pytest.approx(416 / 320)


def test_content_is_preserved_not_just_padding():
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    canvas, meta = CPUPreprocessor().process(frame, target_size=416)
    # somewhere in the middle of the canvas the real (resized) content should
    # appear, distinct from the gray pad value
    center = canvas[208, 208]
    assert center[0] != LETTERBOX_PAD_VALUE
