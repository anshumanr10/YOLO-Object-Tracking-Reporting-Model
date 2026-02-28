"""Adapts Picamera2 to the same interface as cv2.VideoCapture: read(), release(), get(), isOpened()."""
from typing import Any
import cv2
import picamera2


class PiCameraAdapter:
    def __init__(self) -> None:
        self._cam = picamera2.Picamera2()

        mode = self._cam.sensor_modes[0]  # 1920x1080 in your listing

        config = self._cam.create_preview_configuration(
            main={"size": mode["size"]},
            raw=mode
        )

        self._cam.configure(config)
        self._cam.start()
        self._opened = True

        arr = self._cam.capture_array()
        self._width = int(arr.shape[1]) if arr is not None and arr.size else 0
        self._height = int(arr.shape[0]) if arr is not None and arr.size else 0

        self._fps = -1.0
        try:
            limits = self._cam.camera_controls.get("FrameDurationLimits")
            if limits and len(limits) >= 2:
                dur = limits[2] if len(limits) == 3 and limits[2] else limits[1]
                if dur and dur > 0:
                    self._fps = round(1_000_000.0 / dur, 2)
        except (TypeError, ZeroDivisionError):
            pass


    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Any]:
        if not self._opened:
            return False, None
        try:
            arr = self._cam.capture_array()
            if arr is None or arr.size == 0:
                return False, None
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception:
            return False, None

    def release(self) -> None:
        if getattr(self, "_cam", None) is not None:
            try:
                self._cam.close()  # full teardown: stop() + camera.release() + cleanup
            except Exception:
                pass
            self._cam = None
        self._opened = False

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return getattr(self, "_fps", -1.0)
        return -1.0
