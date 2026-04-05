"""
Picamera2 → BGR numpy frames. Each frame: RGB ``capture_array`` → ``COLOR_RGB2BGR`` (required for OpenCV/YOLO).

No sleeps. Configuration mirrors the usual preview pipeline with capped main stream size.
"""

from __future__ import annotations

from typing import Any, Generator

import cv2
import numpy as np
from picamera2 import Picamera2

_MAX_STREAM_SIZE = (1920, 1080)


def _select_sensor_and_main_size(
    picam2: Any, max_stream_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    modes = getattr(picam2, "sensor_modes", None)
    sensor_size: tuple[int, int] | None = None
    if modes:
        best_mode = max(modes, key=lambda m: int((m.get("size") or (0, 0))[0]) * int((m.get("size") or (0, 0))[1]))
        size = best_mode.get("size")
        if size:
            sensor_size = (int(size[0]), int(size[1]))
    if not sensor_size:
        size = picam2.camera_properties.get("PixelArraySize", (640, 480))
        sensor_size = (int(size[0]), int(size[1]))
    max_w, max_h = max_stream_size
    w, h = sensor_size
    if w <= 0 or h <= 0:
        main_size = (640, 480)
    else:
        r = min(max_w / w, max_h / h, 1.0)
        main_size = (int(w * r), int(h * r))
    return sensor_size, main_size


def _start_picamera(
    camera_index: int,
    sensor_mode_index: int | None,
) -> Any:
    picam2 = Picamera2(camera_index)
    modes = getattr(picam2, "sensor_modes", None) or []
    mode = None
    if sensor_mode_index is not None and 0 <= sensor_mode_index < len(modes):
        mode = modes[sensor_mode_index]

    if mode is not None:
        raw = mode.get("size", (640, 480))
        sensor_size = (int(raw[0]), int(raw[1]))
        w, h = sensor_size
        max_w, max_h = _MAX_STREAM_SIZE
        if w <= 0 or h <= 0:
            main_size = (640, 480)
        else:
            r = min(max_w / w, max_h / h, 1.0)
            main_size = (int(w * r), int(h * r))
    else:
        sensor_size, main_size = _select_sensor_and_main_size(picam2, _MAX_STREAM_SIZE)

    config = picam2.create_preview_configuration(
        main={"size": main_size},
        sensor={"output_size": sensor_size},
    )
    picam2.configure(config)
    picam2.start(show_preview=False)
    return picam2


def bgr_frames(
    camera_index: int = 0,
    *,
    sensor_mode_index: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames from a Picamera2 camera.

    Contract: camera starts successfully; ``stop()`` / ``close()`` run when the generator ends.
    """
    picam2 = _start_picamera(camera_index, sensor_mode_index)
    try:
        while True:
            arr = picam2.capture_array()
            if arr is None or arr.size == 0:
                continue
            yield cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    finally:
        picam2.stop()
        picam2.close()
