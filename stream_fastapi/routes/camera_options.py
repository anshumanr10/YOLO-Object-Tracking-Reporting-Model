from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from pi import picamera2_source

from ..config import CAMERA_INDEX
from ..session import get_camera_track, get_session_id


router = APIRouter()


@router.get("/api/cameras", response_class=JSONResponse)
async def list_cameras() -> JSONResponse:
    """Return available cameras from pi.picamera2_source.list_cameras()."""
    try:
        cameras = picamera2_source.list_cameras()
    except RuntimeError:
        return JSONResponse({"cameras": [], "picamera2_available": False})
    return JSONResponse({"cameras": cameras, "picamera2_available": True})


@router.get("/api/cameras/{camera_index}/modes", response_class=JSONResponse)
async def list_sensor_modes(camera_index: int) -> JSONResponse:
    """Return sensor modes for the given camera from pi.picamera2_source.list_sensor_modes()."""
    try:
        modes = picamera2_source.list_sensor_modes(camera_index)
    except RuntimeError:
        return JSONResponse({"camera_index": camera_index, "modes": [], "picamera2_available": False})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open camera {camera_index}: {e}")
    return JSONResponse({"camera_index": camera_index, "modes": modes, "picamera2_available": True})


@router.get("/api/stream/options", response_class=JSONResponse)
async def get_stream_options(request: Request, response: Response):
    """Return current raw stream options (camera_index, sensor_mode_index) for the session."""
    sid = get_session_id(request, response)
    try:
        track = get_camera_track(sid)
        return JSONResponse(track.get_options())
    except RuntimeError as e:
        if "picamera2" in str(e).lower():
            return JSONResponse(
                {
                    "camera_index": CAMERA_INDEX,
                    "sensor_mode_index": None,
                    "picamera2_available": False,
                }
            )
        raise


@router.patch("/api/stream/options", response_class=JSONResponse)
async def patch_stream_options(request: Request, response: Response):
    """Update raw stream options from request body. Only provided fields are updated."""
    sid = get_session_id(request, response)
    track = get_camera_track(sid)
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    if "camera_index" in body:
        track.set_camera_index(int(body["camera_index"]))
    if "sensor_mode_index" in body:
        v = body["sensor_mode_index"]
        track.set_sensor_mode_index(int(v) if v is not None else None)
    return JSONResponse(track.get_options())

