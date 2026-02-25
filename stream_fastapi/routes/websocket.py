from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from webrtc import handle_webrtc_connection

from pi.picamera2_source import Picamera2Source

from ..config import CAMERA_INDEX


router = APIRouter()


def _video_source_factory():
    return Picamera2Source(camera_index=CAMERA_INDEX)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_webrtc_connection(websocket, track_factory=_video_source_factory)
    except WebSocketDisconnect:
        pass

