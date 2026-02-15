"""
API v1: health check, MJPEG video stream, and MJPEG tracking stream.
/stream opens the video source lazily on first request (app.state.video_source).
/tracking uses tracking_frames() which opens its own video source from config.
"""
import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from yolo import tracker
from yolo.video_source import load_video_source

api_router = APIRouter()
MJPEG_BOUNDARY = "frame"


def _stream_frames(cap):
    """Yield MJPEG chunks from the shared video source."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode(".jpg", frame)
        if jpeg is None:
            continue
        chunk = (
            f"--{MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg.tobytes())}\r\n\r\n"
        ).encode() + jpeg.tobytes() + b"\r\n"
        yield chunk


def _stream_tracking_frames():
    """Yield MJPEG chunks from tracking_frames (draw=True). Opens its own video source from config."""
    for frame, _results, _model in tracker.tracking_frames(draw=True):
        _, jpeg = cv2.imencode(".jpg", frame)
        if jpeg is None:
            continue
        chunk = (
            f"--{MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg.tobytes())}\r\n\r\n"
        ).encode() + jpeg.tobytes() + b"\r\n"
        yield chunk


@api_router.get("/health")
def health():
    """Liveness check. Returns 200 with status ok."""
    return {"status": "ok"}


@api_router.get("/stream")
def stream(request: Request):
    """
    MJPEG live stream from the video source (opened lazily on first request).
    """
    cap = getattr(request.app.state, "video_source", None)
    if cap is None:
        try:
            cap = load_video_source()
            if not cap.isOpened():
                raise HTTPException(status_code=503, detail="Video source failed to open")
            request.app.state.video_source = cap
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Video source not available: {e!s}") from e
    elif not cap.isOpened():
        raise HTTPException(status_code=503, detail="Video source not available")
    return StreamingResponse(
        _stream_frames(cap),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


@api_router.get("/tracking")
def tracking(request: Request):
    """
    MJPEG stream with YOLO tracking (detections drawn on frames). Uses config video source (opens its own).
    """
    # Validate we can get at least one frame before starting the stream (avoids ERR_EMPTY_RESPONSE).
    gen = tracker.tracking_frames(draw=True)
    try:
        first = next(gen)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Tracking failed to start: {e!s}",
        ) from e
    def stream_gen():
        frame, _results, _model = first
        _, jpeg = cv2.imencode(".jpg", frame)
        if jpeg is not None:
            yield (
                f"--{MJPEG_BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg.tobytes())}\r\n\r\n"
            ).encode() + jpeg.tobytes() + b"\r\n"
        for frame, _results, _model in gen:
            _, jpeg = cv2.imencode(".jpg", frame)
            if jpeg is None:
                continue
            yield (
                f"--{MJPEG_BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg.tobytes())}\r\n\r\n"
            ).encode() + jpeg.tobytes() + b"\r\n"
    return StreamingResponse(
        stream_gen(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )
