from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter()


TRACKING_VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tracking stream</title>
</head>
<body>
  <h1>Tracking stream</h1>
  <p id="status">Connecting...</p>
  <video id="video" autoplay playsinline muted style="max-width:100%; min-height:240px; background:#000;"></video>
  <script>
    const status = document.getElementById("status");
    const video = document.getElementById("video");
    const ws = new WebSocket(`ws://${location.host}/ws-tracking`);
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


@router.get("/tracking", response_class=HTMLResponse)
async def tracking_viewer_page():
    return TRACKING_VIEWER_HTML
