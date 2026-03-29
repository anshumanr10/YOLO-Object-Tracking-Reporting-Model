"""
Picamera2-backed video source for YOLO: opens the camera and exposes a
synchronous stream of BGR frames suitable for passing directly to `YOLO.track(...)`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# Maximum (width, height) for the main stream; used to cap resolution while preserving aspect ratio.
_MAX_STREAM_SIZE = (1920, 1080)


def _select_sensor_and_main_size(
    picam2: Any, max_stream_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select sensor mode and main stream size for Picamera2.

    Chooses the largest available sensor mode and computes a main stream size
    that fits within max_stream_size while preserving aspect ratio.

    Args:
        picam2: Configured Picamera2 instance (may have sensor_modes).
        max_stream_size: Maximum (width, height) for the main stream.

    Returns:
        A pair (sensor_size, main_size), each (width, height).
    """
    modes = getattr(picam2, "sensor_modes", None)
    sensor_size: tuple[int, int] | None = None

    if modes:
        try:
            def _area(m: Any) -> int:
                """Return pixel area for a sensor mode dict with a 'size' key."""
                size = m.get("size") or (0, 0)
                return int(size[0]) * int(size[1])

            best_mode = max(modes, key=_area)
            size = best_mode.get("size")
            if size:
                sensor_size = (int(size[0]), int(size[1]))
        except Exception:
            sensor_size = None

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


def picamera2_frame_stream(
    camera_index: int = 0,
    target_fps: int = 30,
    sensor_mode_index: int | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield BGR frames from Picamera2 for use with YOLO.

    Configures Picamera2 sensor and main size (capped by internal max stream size),
    converts captured RGB to BGR numpy arrays for OpenCV/YOLO, and yields frames
    at approximately target_fps. The camera is started on first iteration and
    stopped when the generator is closed or exhausted.

    Args:
        camera_index: Picamera2 camera index (default 0).
        target_fps: Desired frame rate; 0 means no sleep between frames.
        sensor_mode_index: If set, use this index into sensor_modes; otherwise
            the largest available sensor mode is selected.

    Yields:
        np.ndarray: BGR frames (H, W, 3), suitable for YOLO.track() or OpenCV.

    Raises:
        RuntimeError: If picamera2 is not installed.
    """
    if Picamera2 is None:
        raise RuntimeError("picamera2 is not installed")

    picam2: Any | None = None
    try:
        picam2 = Picamera2(camera_index)
        # If a specific sensor mode was requested, honor its size; otherwise
        # fall back to automatic selection.
        sensor_size: tuple[int, int]
        main_size: tuple[int, int]

        modes = getattr(picam2, "sensor_modes", None) or []
        mode = None
        if sensor_mode_index is not None and 0 <= sensor_mode_index < len(modes):
            mode = modes[sensor_mode_index]

        if mode is not None:
            sensor_size = tuple(mode.get("size", (640, 480)))  # type: ignore[assignment]
            w, h = sensor_size
            max_w, max_h = _MAX_STREAM_SIZE
            if w <= 0 or h <= 0:
                main_size = (640, 480)
            else:
                r = min(max_w / w, max_h / h, 1.0)
                main_size = (int(w * r), int(h * r))
        else:
            sensor_size, main_size = _select_sensor_and_main_size(
                picam2, _MAX_STREAM_SIZE
            )

        config = picam2.create_preview_configuration(
            main={"size": main_size},
            sensor={"output_size": sensor_size},
        )
        picam2.configure(config)
        picam2.start(show_preview=False)

        delay = 1.0 / float(target_fps) if target_fps > 0 else 0.0
        while True:
            arr = picam2.capture_array()
            if arr is None or arr.size == 0:
                continue
            try:
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                # If conversion fails, skip this frame
                logger.exception("Failed to convert Picamera2 frame to BGR")
                continue
            yield frame
            if delay > 0.0:
                time.sleep(delay)
    except Exception:
        logger.exception("Error in Picamera2 frame stream")
        raise
    finally:
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass

if __name__ == "__main__":
    # Simple smoke test: read a few frames and print their shape.
    frame_count = 0
    for frame in picamera2_frame_stream():
        frame_count += 1
        print(f"Frame {frame_count}: shape={frame.shape}")
        if frame_count >= 10:
            break

