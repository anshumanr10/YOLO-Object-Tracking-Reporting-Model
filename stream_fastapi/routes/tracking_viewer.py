import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from ..session import (
    apply_tracking_patch,
    apply_tracking_rtsp_url,
    get_session_id,
    get_tracking_rtsp_options_dict,
    get_tracking_track,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@router.get("/tracking", response_class=FileResponse)
async def tracking_viewer_page():
    return FileResponse(_FRONTEND_DIR / "tracking.html")


@router.get("/tracking-rtsp", response_class=FileResponse)
async def tracking_rtsp_viewer_page():
    return FileResponse(_FRONTEND_DIR / "tracking_rtsp.html")


@router.get("/tracking/load-model", response_class=JSONResponse)
async def tracking_load_model(request: Request, response: Response):
    """Load the session's tracking model (call session track's load_model)."""
    sid = get_session_id(request, response)
    logger.info("GET /tracking/load-model: session_id=%s…", sid[:12])
    track = get_tracking_track(sid)
    # Heavy CPU/disk work — avoid blocking the asyncio event loop
    await asyncio.to_thread(track.load_model)
    logger.info("GET /tracking/load-model: done session_id=%s…", sid[:12])
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
    opts["session_id"] = sid
    opts["available_models"] = list(config.models.keys()) if config.models else []
    opts["available_classes"] = list(config.classifications.keys()) if config.classifications else []
    return JSONResponse(opts)


@router.patch("/tracking/options", response_class=JSONResponse)
async def tracking_patch_options(request: Request, response: Response):
    """Update tracking options. Merges into session state and replaces the track (plan A)."""
    sid = get_session_id(request, response)
    body: dict = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}

    if not body:
        track = get_tracking_track(sid)
        opts = dict(track.get_options())
        opts["session_id"] = sid
        return JSONResponse(opts)

    track = apply_tracking_patch(sid, body)
    opts = dict(track.get_options())
    opts["session_id"] = sid
    return JSONResponse(opts)


@router.get("/tracking/rtsp/options", response_class=JSONResponse)
async def tracking_rtsp_get_options(request: Request, response: Response):
    """RTSP page: tuning defaults and model lists without creating a yaml-default capture track."""
    from yolo import config_loader as config

    logger.info("GET /tracking/rtsp/options: start (yolo load_config next)")
    config.load_config()
    logger.info("GET /tracking/rtsp/options: load_config finished")
    sid = get_session_id(request, response)
    logger.info("GET /tracking/rtsp/options: session_id=%s…", sid[:12])
    opts = dict(get_tracking_rtsp_options_dict(sid))
    opts["session_id"] = sid
    opts["available_models"] = list(config.models.keys()) if config.models else []
    opts["available_classes"] = list(config.classifications.keys()) if config.classifications else []
    logger.info(
        "GET /tracking/rtsp/options: ok models=%d classes=%d",
        len(opts["available_models"]),
        len(opts["available_classes"]),
    )
    return JSONResponse(opts)


@router.post("/tracking/rtsp", response_class=JSONResponse)
async def tracking_rtsp_apply(request: Request, response: Response):
    """Set capture to an OpenCV URL (POST JSON ``{ "url": "rtsp://..." }``). Source does not use defaults.yaml."""
    from yolo import config_loader as config

    logger.info("POST /tracking/rtsp: start")
    config.load_config()
    sid = get_session_id(request, response)
    raw: dict = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        raw = await request.json() or {}
    url = raw.get("url")
    extra = {k: v for k, v in raw.items() if k != "url"}
    try:
        track = apply_tracking_rtsp_url(sid, url if isinstance(url, str) else "", extra)
    except ValueError as e:
        logger.warning("POST /tracking/rtsp: bad request %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    logger.info("POST /tracking/rtsp: track updated session_id=%s…", sid[:12])
    opts = dict(track.get_options())
    ref = track.source_spec.capture_ref
    opts["url"] = ref if isinstance(ref, str) else None
    opts["session_id"] = sid
    opts["available_models"] = list(config.models.keys()) if config.models else []
    opts["available_classes"] = list(config.classifications.keys()) if config.classifications else []
    return JSONResponse(opts)
