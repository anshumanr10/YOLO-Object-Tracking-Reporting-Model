"""
Picamera2-backed video source for WebRTC: captures frames in a background
thread and exposes them via an aiortc VideoStreamTrack subclass.
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


def list_cameras() -> list:
    """Return list of camera info dicts from Picamera2.global_camera_info()."""
    if Picamera2 is None:
        raise RuntimeError("picamera2 is not installed")
    info = Picamera2.global_camera_info()
    cameras = []
    for index, cam in enumerate(info or []):
        cameras.append({
            "index": index,
            "model": cam.get("Model", "Unknown"),
            "location": cam.get("Location", "Unknown"),
            "rotation": cam.get("Rotation", "Unknown"),
            "id": cam.get("Id", "Unknown"),
        })
    return cameras


def list_sensor_modes(camera_index: int) -> list:
    """Return sensor modes for the given camera index. Caller must not hold the camera open long."""
    if Picamera2 is None:
        raise RuntimeError("picamera2 is not installed")
    picam2 = Picamera2(camera_index)
    try:
        modes = getattr(picam2, "sensor_modes", None) or []
        out = []
        for idx, mode in enumerate(modes):
            size = mode.get("size", (0, 0))
            fps = float(mode.get("fps", 0))
            bit_depth = int(mode.get("bit_depth", 0))
            fmt = mode.get("format", "")
            fmt_str = str(fmt)
            out.append({
                "index": idx,
                "size": [int(size[0]), int(size[1])],
                "fps": fps,
                "bit_depth": bit_depth,
                "format": fmt_str,
            })
        return out
    finally:
        try:
            picam2.close()
        except Exception:
            pass


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


class Picamera2Source(VideoStreamTrack):
    """
    Video source that feeds frames from Picamera2 into the WebRTC video track.
    Runs the camera with NULL preview and a background thread that captures into a queue.
    """

    kind = "video"

    def __init__(self, camera_index: int, sensor_mode_index: int | None = None) -> None:
        super().__init__()
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        self._camera_index = camera_index
        self._sensor_mode_index = sensor_mode_index
        self._picam2: Optional[Any] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def set_camera_index(self, camera_index: int) -> None:
        self._camera_index = camera_index

    def set_sensor_mode_index(self, sensor_mode_index: int | None) -> None:
        self._sensor_mode_index = sensor_mode_index

    def get_options(self) -> dict:
        return {
            "camera_index": self._camera_index,
            "sensor_mode_index": self._sensor_mode_index,
        }

    def _capture_loop(self) -> None:
        try:
            self._picam2 = Picamera2(self._camera_index)
            sensor_size: tuple[int, int]
            main_size: tuple[int, int]

            # If a specific sensor mode was requested, honor its size and
            # scale main stream to fit within _MAX_STREAM_SIZE while
            # preserving aspect ratio (mirrors record_video() logic).
            modes = getattr(self._picam2, "sensor_modes", None) or []
            mode = None
            if (
                self._sensor_mode_index is not None
                and 0 <= self._sensor_mode_index < len(modes)
            ):
                mode = modes[self._sensor_mode_index]

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
                    self._picam2.close()
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
        # Wait briefly for capture thread to exit and run its cleanup
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
        super().stop()
