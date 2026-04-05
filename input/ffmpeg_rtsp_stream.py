"""
OpenCV FFmpeg backend: RTSP (or any URL FFmpeg understands) → BGR numpy frames.

Latency-oriented defaults: small decode queue, optional FFmpeg nobuffer flags via env
(must be set before ``cv2`` is imported — this module applies them at load time).
No sleeps, no color conversion, no copies beyond what ``read()`` returns.
"""

from __future__ import annotations

import os
from typing import Any, Generator

import numpy as np


def _ensure_opencv_ffmpeg_latency_env() -> None:
    """Set ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` once if unset (before ``import cv2``)."""
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
        return
    # TCP is widely compatible; nobuffer asks FFmpeg to reduce demuxer buffering.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer"


_ensure_opencv_ffmpeg_latency_env()

import cv2  # noqa: E402  — after env so libav picks options


def _open_rtsp_capture(url: str, buffer_size: int) -> Any:
    """FFmpeg ``CAP_FFMPEG`` backend, minimal internal frame buffer."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def bgr_frames(url: str, *, buffer_size: int = 1) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames (``H×W×3`` uint8) from a stream URL.

    Contract: ``url`` opens successfully; ``release()`` runs when the generator ends.
    """
    cap = _open_rtsp_capture(url, buffer_size)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame
    finally:
        cap.release()
