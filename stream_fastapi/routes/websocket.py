from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from webrtc import handle_webrtc_connection

from pi.picamera2_source import Picamera2Source
from pi.tracking_source import TrackingVideoTrack

from ..config import CAMERA_INDEX

from yolo import config_loader as yolo_config


router = APIRouter()


def _video_source_factory():
    return Picamera2Source(camera_index=CAMERA_INDEX)


def _tracking_track_factory():
    yolo_config.load_config()
    source = yolo_config.defaults.get("source") or {}
    source_type = source.get("type") or "PiCamera"
    camera_index = source.get("int", CAMERA_INDEX) if isinstance(source.get("int"), (int, float)) else CAMERA_INDEX
    return TrackingVideoTrack(
        source_type=source_type,
        camera_index=int(camera_index),
    )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_webrtc_connection(websocket, track_factory=_video_source_factory)
    except WebSocketDisconnect:
        pass


@router.websocket("/ws-tracking")
async def websocket_tracking_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_webrtc_connection(websocket, track_factory=_tracking_track_factory)
    except WebSocketDisconnect:
        pass

