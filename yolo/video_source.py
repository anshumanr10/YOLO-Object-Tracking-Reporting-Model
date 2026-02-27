from pathlib import Path
import time
from typing import Any, Dict

import cv2

from . import config_loader as config
from .picam_adapter import PiCameraAdapter


# Load config once and capture default source spec.
config.load_config()
_DEFAULT_SOURCE: Dict[str, Any] = config.defaults["source"]


class VideoSource:
    """Single type returned by load_video_source. Same interface for cv2 and Pi camera: read(), release(), get(), isOpened()."""

    def __init__(self, cap: Any) -> None:
        self._cap = cap

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> tuple[bool, Any]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()

    def get(self, prop: int) -> float:
        return self._cap.get(prop)


def load_video_source(video_source: Dict[str, Any] = _DEFAULT_SOURCE) -> VideoSource:
    src = video_source
    src_type = src["type"]
    src_vals = src.get("values") or {}
    if src_type == "Webcam":
        cap = cv2.VideoCapture(src_vals["int"])
    elif src_type in {"RTMP", "RTSP", "HTML"}:
        cap = cv2.VideoCapture(src_vals["url"])
    elif src_type == "File":
        cap = cv2.VideoCapture(src_vals["path"])
    elif src_type == "PiCamera":
        cap = PiCameraAdapter()
    else:
        raise ValueError(f"Unsupported source type: {src_type}")
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: type={src_type}, values={src_vals}")
    return VideoSource(cap)


if __name__ == "__main__":
    print(f"Running: {Path(__file__).resolve()}")
    cap = load_video_source()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {w}x{h}" + (f", FPS: {fps}" if fps > 0 else ""))
    save_dir = Path("saved_frames")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stream open. Running for 5 seconds, saving frames to {save_dir} ...")
    start = time.perf_counter()
    frame_count = 0
    while time.perf_counter() - start < 5.0:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        path = save_dir / f"frame_{frame_count:06d}.jpg"
        if not cv2.imwrite(str(path), frame):
            print(f"Warning: failed to write {path}")
    cap.release()
    print(f"Stream closed. Saved {frame_count} frames to {save_dir}/")