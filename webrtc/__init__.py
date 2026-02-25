"""
WebRTC (aiortc) package: Picamera2 video track and signaling over WebSocket.
Designed to be used by stream_server or any WebSocket server.
"""

from webrtc.connection import handle_webrtc_connection
from webrtc.track import Picamera2Track

__all__ = ["handle_webrtc_connection", "Picamera2Track"]
