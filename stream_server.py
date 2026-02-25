#!/usr/bin/env python3
"""
FastAPI app: serves a viewer page and WebSocket endpoint for WebRTC signaling.
Uses webrtc_handler (aiortc + Picamera2) for the actual stream.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from webrtc import handle_webrtc_connection, Picamera2Track

app = FastAPI(title="Stream server")

# Caller must supply the video source; no default track.
CAMERA_INDEX = 0


def _stream_track_factory():
    return Picamera2Track(camera_index=CAMERA_INDEX)

VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stream viewer</title>
</head>
<body>
  <h1>Stream viewer</h1>
  <p id="status">Connecting...</p>
  <video id="video" autoplay playsinline muted style="max-width:100%; min-height:240px; background:#000;"></video>
  <script>
    const status = document.getElementById("status");
    const video = document.getElementById("video");
    const ws = new WebSocket(`ws://${location.host}/ws`);
    const pc = new RTCPeerConnection();

    pc.ontrack = (e) => {
      status.textContent = "Stream connected";
      video.srcObject = new MediaStream([e.track]);
      video.play().catch(function() {});
    };
    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === "failed" || pc.iceConnectionState === "closed")
        status.textContent = "Connection " + pc.iceConnectionState;
    };

    ws.onopen = async () => {
      status.textContent = "WebSocket connected, negotiating...";
      try {
        pc.addTransceiver("video", { direction: "recvonly" });
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        ws.send(JSON.stringify({ type: offer.type, sdp: offer.sdp }));
      } catch (e) {
        status.textContent = "Error: " + e.message;
      }
    };

    ws.onmessage = async (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "answer" && msg.sdp) {
        await pc.setRemoteDescription(new RTCSessionDescription(msg));
      } else if (msg.type === "ice" && msg.candidate) {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(msg.candidate));
        } catch (e) {}
      }
    };

    ws.onclose = () => { status.textContent = "WebSocket closed"; };
    ws.onerror = () => { status.textContent = "WebSocket error"; };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def viewer_page():
    return VIEWER_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_webrtc_connection(websocket, track_factory=_stream_track_factory)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
