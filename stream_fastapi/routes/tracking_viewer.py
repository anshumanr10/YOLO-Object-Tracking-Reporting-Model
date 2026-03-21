from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from ..session import get_session_id, get_tracking_track

router = APIRouter()

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@router.get("/tracking", response_class=FileResponse)
async def tracking_viewer_page():
    return FileResponse(_FRONTEND_DIR / "tracking.html")


@router.get("/tracking/load-model", response_class=JSONResponse)
async def tracking_load_model(request: Request, response: Response):
    """Load the session's tracking model (call session track's load_model)."""
    sid = get_session_id(request, response)
    track = get_tracking_track(sid)
    track.load_model()
    return JSONResponse({"status": "ok"})


@router.get("/tracking/warmup", response_class=JSONResponse)
async def tracking_warmup(request: Request, response: Response):
    """Legacy alias for /tracking/load-model."""
    return await tracking_load_model(request, response)


@router.get("/tracking/options", response_class=JSONResponse)
async def tracking_get_options(request: Request, response: Response):
    """Return current tracking options plus available_models and available_classes for UI."""
    from yolo import config_loader as config

    config.load_config()
    sid = get_session_id(request, response)
    track = get_tracking_track(sid)
    opts = dict(track.get_options())
    opts["available_models"] = list(config.models.keys()) if config.models else []
    opts["available_classes"] = list(config.classifications.keys()) if config.classifications else []
    return JSONResponse(opts)


@router.patch("/tracking/options", response_class=JSONResponse)
async def tracking_patch_options(request: Request, response: Response):
    """Update tracking options from request body. Only provided fields are updated."""
    from yolo import config_loader as config

    sid = get_session_id(request, response)
    track = get_tracking_track(sid)
    body: dict = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}

    if "source_type" in body:
        track.set_source_type(str(body["source_type"]))
    if "camera_index" in body:
        v = body["camera_index"]
        track.set_camera_index(v if isinstance(v, str) else int(v))
    if "sensor_mode_index" in body:
        v = body["sensor_mode_index"]
        track.set_sensor_mode_index(int(v) if v is not None else None)
    if "target_fps" in body:
        v = body["target_fps"]
        track.set_fps(int(v) if v is not None else None)
    if "model_key" in body:
        track.set_model_key(body["model_key"] if body["model_key"] is not None else None)
    if "conf" in body:
        v = body["conf"]
        track.set_conf(float(v) if v is not None else None)
    if "persist" in body:
        v = body["persist"]
        track.set_persist(bool(v) if v is not None else None)
    if "tracker" in body:
        track.set_tracker(body["tracker"] if body["tracker"] is not None else None)
    if "classes" in body:
        names: Optional[List[str]] = body["classes"]
        if names is None or not names:
            track.set_classes(None)
        else:
            config.load_config()
            ids = [config.classifications[n] for n in names if n in config.classifications]
            track.set_classes(ids if ids else None)

    return JSONResponse(track.get_options())
