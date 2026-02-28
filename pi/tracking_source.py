"""
Picamera2 + YOLO tracking video source for WebRTC: runs the tracker pipeline
in a background thread and exposes annotated frames via an aiortc VideoStreamTrack.
"""

import asyncio
import logging
import queue
import threading
import time
from collections import Counter
from typing import Any, List, Optional

import cv2
import numpy as np
from av import VideoFrame
from aiortc import VideoStreamTrack

try:
    from aiortc.exceptions import MediaStreamError
except ImportError:
    MediaStreamError = Exception  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# Throttle realtime stats to terminal (seconds between lines)
_STATS_INTERVAL = 1.0


class TrackingVideoTrack(VideoStreamTrack):
    """
    Video source that runs camera -> tracker (YOLO + draw) -> queue -> WebRTC.
    Uses yolo.source_picamera when source_type is PiCamera, else yolo.source_opencv.
    """

    kind = "video"

    def __init__(
        self,
        source_type: str = "PiCamera",
        camera_index: int = 0,
        target_fps: Optional[int] = None,
        model_key: Optional[str] = None,
        conf: Optional[float] = None,
        persist: Optional[bool] = None,
        tracker: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self._source_type = source_type
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._model_key = model_key
        self._conf = conf
        self._persist = persist
        self._tracker = tracker
        self._classes = classes
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._pipeline_ready = False

    def _tracking_loop(self) -> None:
        try:
            from yolo.tracker import tracking_frames
            from yolo import model as model_setup

            print("[tracking] pipeline starting (loading model & camera)...", flush=True)

            # Preload model so first frame is not delayed by 30–60s on Pi
            model_key = self._model_key if self._model_key is not None else getattr(
                model_setup, "_DEFAULT_MODEL", "yolov8n"
            )
            model_setup.load_model(model_key)

            if self._source_type == "PiCamera":
                from yolo.source_picamera import picamera2_frame_stream

                stream_kwargs: dict = {"camera_index": self._camera_index}
                if self._target_fps is not None:
                    stream_kwargs["target_fps"] = self._target_fps
                frame_stream = picamera2_frame_stream(**stream_kwargs)
            else:
                from yolo.source_opencv import cv_frame_stream

                capture = cv2.VideoCapture(self._camera_index)
                stream_kwargs: dict = {"capture": capture}
                if self._target_fps is not None:
                    stream_kwargs["target_fps"] = self._target_fps
                frame_stream = cv_frame_stream(**stream_kwargs)

            track_kwargs: dict = {"frame_stream": frame_stream, "draw": True}
            if self._model_key is not None:
                track_kwargs["model_key"] = self._model_key
            if self._conf is not None:
                track_kwargs["conf"] = self._conf
            if self._persist is not None:
                track_kwargs["persist"] = self._persist
            if self._tracker is not None:
                track_kwargs["tracker"] = self._tracker
            if self._classes is not None:
                track_kwargs["classes"] = self._classes

            last_stats_time = time.monotonic()
            frame_count = 0
            for frame, results, model in tracking_frames(**track_kwargs):
                if self._stop.is_set():
                    break
                frame_count += 1
                # Realtime detection stats to terminal (throttled)
                now = time.monotonic()
                if now - last_stats_time >= _STATS_INTERVAL:
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        names = getattr(model, "names", {})
                        counts = Counter(
                            names.get(int(b.cls.item()), str(int(b.cls.item())))
                            for b in r.boxes
                        )
                        parts = ", ".join(f"{n}: {c}" for n, c in sorted(counts.items()))
                        total = len(r.boxes)
                        print(f"[tracking] frame {frame_count} | {parts} | {total} detections", flush=True)
                    else:
                        print(f"[tracking] frame {frame_count} | 0 detections", flush=True)
                    last_stats_time = now
                # frame is BGR from OpenCV; WebRTC expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                try:
                    self._frame_queue.put_nowait(rgb)
                except queue.Full:
                    pass
        except Exception as e:
            logger.exception("Tracking loop error: %s", e)
            print(f"[tracking] ERROR: {e}", flush=True)

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    async def recv(self) -> VideoFrame:
        self._ensure_started()
        # Only start taking frames after the pipeline is ready (camera on, model loaded, first frame produced)
        if not self._pipeline_ready:
            deadline = time.monotonic() + 90.0
            while (
                self._frame_queue.qsize() == 0
                and self._thread is not None
                and self._thread.is_alive()
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.5)
            if self._frame_queue.qsize() == 0:
                raise MediaStreamError("Tracking pipeline did not become ready")
            self._pipeline_ready = True
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
