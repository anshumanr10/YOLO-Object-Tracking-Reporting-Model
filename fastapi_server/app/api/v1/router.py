import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MJPEG_BOUNDARY = "frame"

api_router = APIRouter()


class TrackRequest(BaseModel):
    """Optional body for POST /track. Omit to use config default source."""

    video_source: Optional[Dict[str, Any]] = Field(
        default=None,
        description='e.g. {"type": "File", "values": {"path": "/path/to/video.mp4"}}',
    )
    max_frames: Optional[int] = Field(default=None, ge=1, description="Limit frames (recommended for live sources)")
    write_summary_file: bool = Field(default=False, description="Write detections_summary.txt on server")


def _frames_to_mjpeg_stream(
    video_source_spec=None,
    conf=None,
    max_frames=None,
    scale: Optional[float] = None,
    jpeg_quality: Optional[int] = None,
):
    """Consume yolo.tracker.tracking_frames() and yield MJPEG parts. scale<1 and jpeg_quality reduce bandwidth."""
    from yolo.tracker import tracking_frames

    encode_opts = []
    if jpeg_quality is not None and 1 <= jpeg_quality <= 100:
        encode_opts.append(cv2.IMWRITE_JPEG_QUALITY)
        encode_opts.append(int(jpeg_quality))

    for frame, _, _ in tracking_frames(
        video_source_spec=video_source_spec,
        conf=conf,
        max_frames=max_frames,
        draw=True,
    ):
        if scale is not None and 0 < scale < 1 and frame is not None:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, jpeg = cv2.imencode(".jpg", frame, encode_opts if encode_opts else [])
        jpeg_bytes = jpeg.tobytes()
        part = (
            f"--{MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
        ).encode() + jpeg_bytes + b"\r\n"
        yield part


@api_router.get("/health")
def health_check():
    return {"status": "ok"}


@api_router.post("/track", response_model=Dict[str, Any])
def track(req: TrackRequest = Body(default_factory=TrackRequest)):
    """
    Run YOLO tracking on a video source. Returns detection/tracking stats as JSON.

    Send an empty body `{}` to use config default source, or set `video_source`, `max_frames`, etc.
    """
    try:
        from yolo.tracker import run_tracking
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"YOLO tracker not available: {e}") from e

    try:
        stats = run_tracking(
            video_source_spec=req.video_source,
            max_frames=req.max_frames,
            write_summary_file=req.write_summary_file,
            show_display=False,
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@api_router.post("/track/upload", response_model=Dict[str, Any])
async def track_upload(
    file: UploadFile = File(...),
    max_frames: Optional[int] = None,
    write_summary_file: bool = False,
):
    """
    Upload a video file, run YOLO tracking on it, return stats. File is discarded after processing.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Expected a video file")

    try:
        from yolo.tracker import run_tracking
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"YOLO tracker not available: {e}") from e

    suffix = Path(file.filename or "video").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stats = run_tracking(
            video_source_spec={"type": "File", "values": {"path": tmp_path}},
            max_frames=max_frames,
            write_summary_file=write_summary_file,
            show_display=False,
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@api_router.get("/stream")
def stream_mjpeg(
    max_frames: Optional[int] = None,
    scale: Optional[float] = None,
    jpeg_quality: Optional[int] = None,
):
    """
    MJPEG live stream of YOLO tracking using the config default source.

    Query params: max_frames, scale (e.g. 0.5 for half size), jpeg_quality (1-100).
    Use scale and jpeg_quality to reduce bandwidth when streaming via ngrok.
    """
    try:
        gen = _frames_to_mjpeg_stream(
            video_source_spec=None,
            conf=None,
            max_frames=max_frames,
            scale=scale,
            jpeg_quality=jpeg_quality,
        )
        return StreamingResponse(
            gen,
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
