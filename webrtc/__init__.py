"""
WebRTC (aiortc) package: signaling over WebSocket and connection handling.
Source-agnostic: callers supply a track factory that returns a VideoStreamTrack.
"""

from webrtc.connection import handle_webrtc_connection

__all__ = ["handle_webrtc_connection"]
