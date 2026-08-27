"""Synthetic video source generation for experiments that don't have real
footage supplied. Never writes into the repo -- callers pass a directory
(a tempdir by default), and CLAUDE.md's rule against committing video is
satisfied by that directory never being under version control, not by any
special-casing here.
"""
from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np


def generate_synthetic_sources(
    output_dir: str,
    count: int,
    n_frames: int = 60,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
    seed: int = 0,
) -> List[str]:
    """Writes `count` short synthetic clips into output_dir and returns
    their paths, for experiments run without real video on hand."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(count):
        path = os.path.join(output_dir, f"synthetic_{i}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for _ in range(n_frames):
            writer.write(rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8))
        writer.release()
        paths.append(path)
    return paths


def resolve_sources(count: int, output_dir: str, source_dir: str = None, **synthetic_kwargs) -> List[str]:
    """Real files from source_dir (round-robin reused up to `count`) if
    given, else freshly generated synthetic clips written into output_dir."""
    if source_dir is None:
        return generate_synthetic_sources(output_dir, count, **synthetic_kwargs)

    files = sorted(
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not files:
        raise ValueError(f"no video files found in {source_dir!r}")
    return [files[i % len(files)] for i in range(count)]
