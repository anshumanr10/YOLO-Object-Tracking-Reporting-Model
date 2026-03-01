"""
Per-session state for Option B: one tracking track and one camera track per session.
Session is identified by a cookie (session_id). No login yet; can be layered later.
"""
import uuid
from typing import Any, Dict, Optional

from fastapi import Request, Response

from .config import CAMERA_INDEX

_COOKIE_NAME = "session_id"
_SESSIONS: Dict[str, Dict[str, Any]] = {}


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
    """Read session_id from WebSocket Cookie header. Returns existing or new id (cookie not set on WS)."""
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


def get_tracking_track(session_id: str):
    """Get or create the TrackingVideoTrack for this session. Returns the instance."""
    from pi.tracking_source import TrackingVideoTrack
    from yolo import config_loader as config

    config.load_config()
    session = get_session(session_id)
    if "tracking" not in session or session["tracking"] is None:
        source_cfg = config.defaults.get("source") or {}
        default_source_type = source_cfg.get("type") or "PiCamera"
        default_camera_index = CAMERA_INDEX
        if isinstance(source_cfg.get("camera_index"), (int, float)):
            default_camera_index = int(source_cfg["camera_index"])
        session["tracking"] = TrackingVideoTrack(
            source_type=default_source_type,
            camera_index=default_camera_index,
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
