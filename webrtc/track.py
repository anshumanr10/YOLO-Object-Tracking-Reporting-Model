"""
Picamera2 video track for WebRTC: captures frames in a background thread and
feeds them to aiortc's VideoStreamTrack.
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Optional

import numpy as np
from av import VideoFrame
from aiortc import VideoStreamTrack

try:
    from aiortc.exceptions import MediaStreamError
except ImportError:
    MediaStreamError = Exception  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # type: ignore[misc, assignment]

_MAX_STREAM_SIZE = (1920, 1080)


def select_sensor_and_main_size(picam2: Any, max_stream_size: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Given a Picamera2 instance and a maximum stream size (width, height),
    select the largest available sensor mode and compute a main stream size
    that fits within max_stream_size while preserving aspect ratio.
    """
    modes = getattr(picam2, "sensor_modes", None)
    sensor_size: tuple[int, int] | None = None

    if modes:
        try:
            def _area(m: Any) -> int:
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


class Picamera2Track(VideoStreamTrack):
    """
    Video track that feeds frames from Picamera2 into the WebRTC stream.
    Runs the camera with NULL preview and a background thread that captures into a queue.
    """

    kind = "video"

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        self._camera_index = camera_index
        self._picam2: Optional[Any] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def _capture_loop(self) -> None:
        try:
            self._picam2 = Picamera2(self._camera_index)
            sensor_size, main_size = select_sensor_and_main_size(
                self._picam2, _MAX_STREAM_SIZE
            )

            config = self._picam2.create_preview_configuration(
                main={"size": main_size},
                sensor={"output_size": sensor_size},
            )
            self._picam2.configure(config)
            self._picam2.start(show_preview=False)
            while not self._stop.is_set():
                arr = self._picam2.capture_array()
                if arr is None:
                    continue
                try:
                    self._frame_queue.put_nowait(arr)
                except queue.Full:
                    pass
                time.sleep(1 / 30)
        except Exception as e:
            logger.exception("Picamera2 capture loop error: %s", e)
        finally:
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                except Exception:
                    pass
                self._picam2 = None

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    async def recv(self) -> VideoFrame:
        self._ensure_started()
        if self._thread is not None and not self._frame_queue.qsize():
            await asyncio.sleep(0.5)
        try:
            arr = await asyncio.to_thread(
                lambda: self._frame_queue.get(timeout=5.0)
            )
        except queue.Empty:
            raise MediaStreamError("No frame from camera")
        if arr is None:
            raise MediaStreamError("Camera stopped")
        if arr.shape[-1] == 4:
            arr = np.ascontiguousarray(arr[:, :, :3])
        else:
            arr = np.ascontiguousarray(arr)
        frame = VideoFrame.from_ndarray(arr, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            self._picam2 = None
        super().stop()
