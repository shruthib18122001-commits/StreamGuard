import pytest

from streamguard.decode.cpu import CPUDecoder


def test_read_frame_before_open_raises():
    decoder = CPUDecoder()
    with pytest.raises(RuntimeError):
        decoder.read_frame()


def test_open_bad_source_raises():
    decoder = CPUDecoder()
    with pytest.raises(RuntimeError):
        decoder.open("/definitely/not/a/real/video.mp4")


def test_reads_all_frames_then_none(synthetic_video):
    path, n_frames, width, height = synthetic_video
    decoder = CPUDecoder()
    decoder.open(path)
    try:
        count = 0
        while True:
            frame = decoder.read_frame()
            if frame is None:
                break
            assert frame.shape == (height, width, 3)
            count += 1
        assert count == n_frames
    finally:
        decoder.close()


def test_close_is_idempotent(synthetic_video):
    path, _, _, _ = synthetic_video
    decoder = CPUDecoder()
    decoder.open(path)
    decoder.close()
    decoder.close()  # must not raise
