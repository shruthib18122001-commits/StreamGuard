import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    """A short synthetic clip written to a pytest tmp dir — never committed,
    never downloaded. Returns (path, n_frames, width, height)."""
    width, height, fps, n_frames = 320, 240, 10, 15
    path = str(tmp_path / "synthetic.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return path, n_frames, width, height
