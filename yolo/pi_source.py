"""
Picamera2-backed video source for YOLO: opens the camera with the same
configuration logic as `pi/picamera2_source.py`, but exposes a synchronous
stream of frames suitable for passing directly to `YOLO.track(...)`.
"""

from __future__ import annotations

import logging
import time
from typing import Generator, Any

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # type: ignore[misc, assignment]

from pi.picamera2_source import select_sensor_and_main_size

logger = logging.getLogger(__name__)

# Keep stream size consistent with `picamera2_source.py`
_MAX_STREAM_SIZE = (1920, 1080)


def picamera2_frame_stream(
    camera_index: int = 0,
    target_fps: int = 30,
) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames from Picamera2 for use with YOLO.

    - Configures Picamera2 sensor/main size via select_sensor_and_main_size,
      matching the WebRTC stream configuration.
    - Converts captured RGB arrays to BGR numpy arrays for OpenCV / YOLO.
    """
    if Picamera2 is None:
        raise RuntimeError("picamera2 is not installed")

    picam2: Any | None = None
    try:
        picam2 = Picamera2(camera_index)
        sensor_size, main_size = select_sensor_and_main_size(picam2, _MAX_STREAM_SIZE)

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

