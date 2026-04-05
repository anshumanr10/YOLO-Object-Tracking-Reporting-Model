"""Stream helpers: RTSP (FFmpeg), webcam, Picamera2 → BGR frame generators."""

from .ffmpeg_rtsp_stream import bgr_frames as rtsp_bgr_frames
from .webcam_cv_stream import bgr_frames as webcam_bgr_frames

# Same as ``ffmpeg_rtsp_stream.bgr_frames`` (RTSP URL); stable package import.
bgr_frames = rtsp_bgr_frames

__all__ = ["bgr_frames", "rtsp_bgr_frames", "webcam_bgr_frames"]

try:
    from .picamera2_stream import bgr_frames as picamera_bgr_frames
except ImportError:
    pass
else:
    __all__.append("picamera_bgr_frames")
