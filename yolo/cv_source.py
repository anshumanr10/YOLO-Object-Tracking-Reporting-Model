"""
cv_source
---------

Generator-based video source for any OpenCV-compatible capture object.

This mirrors `yolo/pi_source.py`, but instead of configuring Picamera2
directly, it consumes an existing cv2-compatible source (e.g.
`cv2.VideoCapture`, RTSP/RTMP stream, file, or a custom adapter with a
`read()` / `isOpened()` / `release()` API).

The frames yielded are BGR `numpy.ndarray` images suitable for passing
directly to `YOLO.track(...)`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

import numpy as np

logger = logging.getLogger(__name__)


def cv_frame_stream(
    capture: Any,
    target_fps: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames from an existing OpenCV-like capture object.

    Parameters
    ----------
    capture:
        An object implementing at least:
          - isOpened() -> bool
          - read() -> tuple[bool, frame]
          - release() -> None
        Typically a `cv2.VideoCapture` or a compatible adapter.

    target_fps:
        If provided and > 0, a simple sleep is used between frames to
        approximate the desired frame rate.
    """
    delay = 1.0 / float(target_fps) if target_fps and target_fps > 0 else 0.0

    if hasattr(capture, "isOpened") and not capture.isOpened():
        logger.warning("cv_frame_stream: capture is not opened")

    try:
        while True:
            try:
                ret, frame = capture.read()
            except Exception:
                logger.exception("cv_frame_stream: capture.read() failed")
                break

            if not ret or frame is None:
                break

            yield frame

            if delay > 0.0:
                time.sleep(delay)
    finally:
        # Best-effort cleanup; some custom capture objects may not implement release().
        try:
            release = getattr(capture, "release", None)
            if callable(release):
                release()
        except Exception:
            logger.exception("cv_frame_stream: capture.release() failed")


if __name__ == "__main__":
    import cv2  # type: ignore[import]

    cap = cv2.VideoCapture(0)
    frame_count = 0
    for frame in cv_frame_stream(cap, target_fps=30):
        frame_count += 1
        print(f"Frame {frame_count}: shape={getattr(frame, 'shape', None)}")
        if frame_count >= 10:
            break

