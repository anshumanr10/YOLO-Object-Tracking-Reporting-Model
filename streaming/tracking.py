"""
Threaded YOLO tracking video track for WebRTC.
This module is device-agnostic: source resolution and capture setup are delegated
to input_sources.resolver.
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

from input_sources.resolver import (
    InputSourceSpec,
    api_options_from_spec,
    make_frame_stream,
)

try:
    from aiortc.exceptions import MediaStreamError
except ImportError:
    MediaStreamError = Exception  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# Throttle realtime stats to terminal (seconds between lines)
_STATS_INTERVAL = 1.0


class TrackingVideoTrack(VideoStreamTrack):
    """Run frame stream -> tracker -> queue -> WebRTC VideoStreamTrack."""

    kind = "video"

    def __init__(
        self,
        source_spec: InputSourceSpec,
        target_fps: Optional[int] = None,
        model_key: Optional[str] = None,
        conf: Optional[float] = None,
        persist: Optional[bool] = None,
        tracker: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self._source_spec = source_spec
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
        self._model: Any = None

    @property
    def source_spec(self) -> InputSourceSpec:
        return self._source_spec

    def load_model(self) -> None:
        """Load the current model (from config default or set_model_key) and cache on this instance."""
        from yolo import config_loader as config
        from ultralytics import YOLO

        config.load_config()
        model_key = self._model_key if self._model_key is not None else config.defaults["model"]
        self._model = YOLO(config.models[model_key]["weights"])

    def get_options(self) -> dict:
        """Return current options for API/UI. Classes are returned as class names when available."""
        from yolo import config_loader as config

        config.load_config()
        id_to_name = {v: k for k, v in (config.classifications or {}).items()}
        class_names = None
        if self._classes:
            class_names = [id_to_name.get(i, str(i)) for i in self._classes]
        source_opts = api_options_from_spec(self._source_spec)
        return {
            **source_opts,
            "target_fps": self._target_fps,
            "model_key": self._model_key,
            "conf": self._conf,
            "persist": self._persist,
            "tracker": self._tracker,
            "classes": class_names,
        }

    def _tracking_loop(self) -> None:
        try:
            from yolo import config_loader as config
            from yolo.tracker import tracking_frames, get_target_class_ids

            config.load_config()
            print("[tracking] pipeline starting (loading model & camera)...", flush=True)

            if self._model is None:
                model_key = self._model_key if self._model_key is not None else config.defaults["model"]
                from ultralytics import YOLO

                self._model = YOLO(config.models[model_key]["weights"])

            frame_stream = make_frame_stream(self._source_spec, target_fps=self._target_fps)

            track_kwargs: dict = {
                "frame_stream": frame_stream,
                "draw": True,
                "model": self._model,
                "model_key": self._model_key if self._model_key is not None else config.defaults["model"],
                "conf": float(self._conf) if self._conf is not None else float(config.defaults.get("conf", 0.5)),
                "persist": self._persist if self._persist is not None else bool(config.defaults.get("tracking", True)),
                "tracker": self._tracker or "bytetrack.yaml",
                "classes": self._classes if self._classes is not None else get_target_class_ids(),
            }

            last_stats_time = time.monotonic()
            frame_count = 0
            for frame, results, model in tracking_frames(**track_kwargs):
                if self._stop.is_set():
                    break
                frame_count += 1
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
            arr = await asyncio.to_thread(lambda: self._frame_queue.get(timeout=5.0))
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
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=5.0)
            except Exception:
                pass
        super().stop()
