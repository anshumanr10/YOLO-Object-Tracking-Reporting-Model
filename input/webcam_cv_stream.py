"""
Local webcam (or V4L device index) via OpenCV ``VideoCapture`` → BGR numpy frames.

No FFmpeg env; minimal decode queue where the backend supports ``CAP_PROP_BUFFERSIZE``.
No sleeps, no color conversion beyond what ``read()`` returns.
"""

from __future__ import annotations

from typing import Any, Generator

import cv2
import numpy as np


def _open_webcam(device_index: int, buffer_size: int, api_preference: int | None) -> Any:
    if api_preference is None:
        cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(device_index, api_preference)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def bgr_frames(
    device_index: int = 0,
    *,
    buffer_size: int = 1,
    api_preference: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames (``H×W×3`` uint8) from a host camera device index.

    Contract: device opens successfully; ``release()`` runs when the generator ends.
    ``api_preference`` is an OpenCV ``VideoCaptureAPIs`` value (e.g. ``cv2.CAP_V4L2``) or ``None`` for default.
    """
    cap = _open_webcam(device_index, buffer_size, api_preference)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame
    finally:
        cap.release()
