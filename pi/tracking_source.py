"""
Picamera2 + YOLO tracking video source for WebRTC: runs the tracker_new pipeline
in a background thread and exposes annotated frames via an aiortc VideoStreamTrack.
"""

import asyncio
import logging
import queue
import threading
from typing import Any, List, Optional

import cv2
import numpy as np
from av import VideoFrame
from aiortc import VideoStreamTrack

try:
    from aiortc.exceptions import MediaStreamError
except ImportError:
    MediaStreamError = Exception  # type: ignore[misc, assignment]

from yolo.tracker_new import (
    _DEFAULT_CONF,
    _DEFAULT_MODEL_KEY,
    _DEFAULT_PERSIST,
    _DEFAULT_TRACKER,
    get_target_class_ids,
)
from yolo.model import _DEFAULT_FPS

logger = logging.getLogger(__name__)


class TrackingVideoTrack(VideoStreamTrack):
    """
    Video source that runs camera -> tracker_new (YOLO + draw) -> queue -> WebRTC.
    Uses yolo.pi_source.picamera2_frame_stream and yolo.tracker_new.tracking_frames.
    """

    kind = "video"

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: Optional[int] = None,
        model_key: Optional[str] = None,
        conf: Optional[float] = None,
        persist: Optional[bool] = None,
        tracker: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._target_fps = target_fps if target_fps is not None else _DEFAULT_FPS
        self._model_key = model_key if model_key is not None else _DEFAULT_MODEL_KEY
        self._conf = conf if conf is not None else _DEFAULT_CONF
        self._persist = persist if persist is not None else _DEFAULT_PERSIST
        self._tracker = tracker if tracker is not None else _DEFAULT_TRACKER
        self._classes = classes if classes is not None else get_target_class_ids()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def _tracking_loop(self) -> None:
        try:
            from yolo.pi_source import picamera2_frame_stream
            from yolo.tracker_new import tracking_frames

            frame_stream = picamera2_frame_stream(
                camera_index=self._camera_index,
                target_fps=self._target_fps,
            )
            for frame, _results, _model in tracking_frames(
                frame_stream=frame_stream,
                model_key=self._model_key,
                conf=self._conf,
                persist=self._persist,
                tracker=self._tracker,
                classes=self._classes,
                draw=True,
            ):
                if self._stop.is_set():
                    break
                # frame is BGR from OpenCV; WebRTC expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                try:
                    self._frame_queue.put_nowait(rgb)
                except queue.Full:
                    pass
        except Exception as e:
            logger.exception("Tracking loop error: %s", e)

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
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
            raise MediaStreamError("No frame from tracking pipeline")
        if arr is None:
            raise MediaStreamError("Tracking pipeline stopped")
        frame = VideoFrame.from_ndarray(arr, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        self._stop.set()
        super().stop()
