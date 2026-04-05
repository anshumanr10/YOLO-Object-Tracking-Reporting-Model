"""
Per-session state for Option B: one tracking track and one camera track per session.
Session is identified by a cookie (session_id). No login yet; can be layered later.

Tracking input and tuning options are merged here; ``streaming.tracking.TrackingVideoTrack`` is
immutable (no setters). PATCH replaces the track instance (plan A).
"""
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Request, Response

from .config import CAMERA_INDEX

_COOKIE_NAME = "session_id"
_SESSIONS: Dict[str, Dict[str, Any]] = {}

_TRACKING_OPTION_KEYS = (
    "source_type",
    "camera_index",
    "sensor_mode_index",
    "url",
    "values",
    "target_fps",
    "model_key",
    "conf",
    "persist",
    "tracker",
    "classes",
)


def get_session_id(request: Request, response: Optional[Response] = None) -> str:
    """Return session ID from cookie, or create one and set cookie on response if provided."""
    sid = request.cookies.get(_COOKIE_NAME)
    if sid:
        return sid
    sid = uuid.uuid4().hex
    if response is not None:
        response.set_cookie(_COOKIE_NAME, sid, httponly=True, samesite="lax")
    return sid


def get_session_id_from_websocket(websocket) -> str:
    """
    Resolve session id for WebSocket connections.

    Prefer ``session_id`` query param (set by the frontend from GET /tracking/options) because
    the tracking ``session_id`` cookie is HttpOnly and may not be visible to JS; some clients
    also omit cookies on WebSocket handshakes, which previously created a new session per WS
    and ignored PATCHed input (e.g. RTSP).
    """
    try:
        q = websocket.query_params.get("session_id")
        if isinstance(q, str) and q.strip():
            return q.strip()
    except Exception:
        pass
    for name, value in websocket.scope.get("headers", []):
        if name == b"cookie":
            cookie_str = value.decode("latin-1")
            for part in cookie_str.split(";"):
                part = part.strip()
                if part.startswith(_COOKIE_NAME + "="):
                    return part.split("=", 1)[1].strip()
            break
    return uuid.uuid4().hex


def get_session(session_id: str) -> Dict[str, Any]:
    """Return the session state dict for the given session ID."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {}
    return _SESSIONS[session_id]


def _default_tracking_options_dict() -> Dict[str, Any]:
    """Baseline options before any session track exists (from defaults.yaml)."""
    from input_sources.resolver import api_options_from_spec, resolve_source_spec
    from yolo import config_loader as config

    config.load_config()
    source_spec = resolve_source_spec(
        config.defaults.get("source") or {},
        default_device_index=CAMERA_INDEX,
    )
    opts = api_options_from_spec(source_spec)
    opts.update(_tuning_only_defaults_from_yaml())
    return opts


def _tuning_only_defaults_from_yaml() -> Dict[str, Any]:
    """Model / FPS / conf / tracker defaults from yaml — no input source."""
    from yolo import config_loader as config

    config.load_config()
    return {
        "target_fps": int(config.defaults.get("fps", 30)),
        "model_key": config.defaults.get("model"),
        "conf": float(config.defaults.get("conf", 0.5)),
        "persist": bool(config.defaults.get("tracking", True)),
        "tracker": "bytetrack.yaml",
        "classes": None,
    }


def _replace_tracking_track_from_merged(session_id: str, merged: Dict[str, Any]):
    """Stop existing track and create ``TrackingVideoTrack`` from a fully merged options dict."""
    from streaming.tracking import TrackingVideoTrack
    from input_sources.resolver import source_spec_from_flat
    from yolo import config_loader as config

    config.load_config()
    session = get_session(session_id)
    old = session.get("tracking")

    class_ids: Optional[List[int]] = None
    if merged.get("classes") is not None and isinstance(merged["classes"], list):
        names = [str(n) for n in merged["classes"] if n]
        class_ids = [config.classifications[n] for n in names if n in config.classifications]
        class_ids = class_ids if class_ids else None

    target_fps = merged.get("target_fps")
    target_fps_i = int(target_fps) if target_fps is not None else None
    conf = merged.get("conf")
    conf_f = float(conf) if conf is not None else None

    spec = source_spec_from_flat(merged, default_device_index=CAMERA_INDEX)

    if old is not None:
        try:
            old.stop()
        except Exception:
            pass

    session["tracking"] = TrackingVideoTrack(
        source_spec=spec,
        target_fps=target_fps_i,
        model_key=merged.get("model_key"),
        conf=conf_f,
        persist=merged.get("persist"),
        tracker=merged.get("tracker"),
        classes=class_ids,
    )
    return session["tracking"]


def apply_tracking_patch(session_id: str, body: Dict[str, Any]):
    """
    Merge PATCH body into current options, stop the old track, create a new TrackingVideoTrack.
    All device/source interpretation uses input_sources.resolver.
    """
    from yolo import config_loader as config

    config.load_config()
    session = get_session(session_id)
    old = session.get("tracking")
    base = old.get_options() if old is not None else _default_tracking_options_dict()

    merged: Dict[str, Any] = dict(base)
    for k in _TRACKING_OPTION_KEYS:
        if k in body:
            merged[k] = body[k]

    return _replace_tracking_track_from_merged(session_id, merged)


def apply_tracking_rtsp_url(session_id: str, url: str, body: Optional[Dict[str, Any]] = None):
    """
    Set tracking input to an OpenCV URL (RTSP, etc.) without using ``defaults.yaml`` for the source.
    Tunings come from the existing track if any, else yaml tuning-only defaults.
    Optional ``body`` may override model_key, target_fps, conf, classes, etc.
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("url is required")

    old = get_session(session_id).get("tracking")
    if old is not None:
        merged = dict(old.get_options())
    else:
        merged = dict(_tuning_only_defaults_from_yaml())

    merged["source_type"] = "OpenCV"
    merged["url"] = url
    merged["sensor_mode_index"] = None
    merged["camera_index"] = 0

    if body:
        for k in _TRACKING_OPTION_KEYS:
            if k in body:
                merged[k] = body[k]

    return _replace_tracking_track_from_merged(session_id, merged)


def get_tracking_rtsp_options_dict(session_id: str) -> Dict[str, Any]:
    """
    Options for the RTSP page: existing track, or yaml tuning defaults with no capture until POST.
    Does not create a yaml-default ``TrackingVideoTrack``.
    """
    session = get_session(session_id)
    old = session.get("tracking")
    if old is not None:
        opts = dict(old.get_options())
        ref = old.source_spec.capture_ref
        opts["url"] = ref if isinstance(ref, str) else None
        return opts
    o = dict(_tuning_only_defaults_from_yaml())
    o["source_type"] = "OpenCV"
    o["camera_index"] = None
    o["sensor_mode_index"] = None
    o["url"] = None
    return o


def get_tracking_track(session_id: str):
    """Get or create the TrackingVideoTrack for this session. Returns the instance."""
    from streaming.tracking import TrackingVideoTrack
    from input_sources.resolver import resolve_source_spec
    from yolo import config_loader as config

    config.load_config()
    session = get_session(session_id)
    if "tracking" not in session or session["tracking"] is None:
        source_spec = resolve_source_spec(
            config.defaults.get("source") or {},
            default_device_index=CAMERA_INDEX,
        )
        session["tracking"] = TrackingVideoTrack(
            source_spec=source_spec,
            target_fps=int(config.defaults.get("fps", 30)),
            model_key=config.defaults.get("model"),
            conf=float(config.defaults.get("conf", 0.5)),
            persist=bool(config.defaults.get("tracking", True)),
            tracker="bytetrack.yaml",
            classes=None,  # will use config default via get_target_class_ids
        )
    return session["tracking"]


def get_camera_track(session_id: str):
    """Get or create the Picamera2Source for this session. Returns the instance."""
    from pi.picamera2_source import Picamera2Source

    session = get_session(session_id)
    if "camera" not in session or session["camera"] is None:
        session["camera"] = Picamera2Source(camera_index=CAMERA_INDEX, sensor_mode_index=None)
    return session["camera"]
