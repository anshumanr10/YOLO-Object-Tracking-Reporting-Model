from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from webrtc import handle_webrtc_connection

from ..session import get_camera_track, get_session_id_from_websocket, get_tracking_track


router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        sid = get_session_id_from_websocket(websocket)
        track = get_camera_track(sid)
        await handle_webrtc_connection(websocket, track_factory=lambda: track)
    except WebSocketDisconnect:
        pass


@router.websocket("/ws-tracking")
async def websocket_tracking_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        sid = get_session_id_from_websocket(websocket)
        track = get_tracking_track(sid)
        await handle_webrtc_connection(websocket, track_factory=lambda: track)
    except WebSocketDisconnect:
        pass

