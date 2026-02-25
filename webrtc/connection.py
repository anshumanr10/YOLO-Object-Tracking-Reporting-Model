"""
WebRTC connection workflow: run signaling over a WebSocket and stream a video track.
Uses track factory for testability and replaceability.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription

from webrtc.ice import ice_candidate_to_dict, parse_ice_candidate_from_message
from webrtc.signaling import build_answer_message, build_ice_message, parse_message

logger = logging.getLogger(__name__)


async def handle_webrtc_connection(
    websocket: Any,
    track_factory: Callable[[], Any],
) -> None:
    """
    Run WebRTC signaling over the given WebSocket and stream the track.

    Expects JSON messages: {"type": "offer", "sdp": "..."} and {"type": "ice", "candidate": {...}}.
    Sends back: {"type": "answer", "sdp": "..."} and {"type": "ice", "candidate": {...}}.

    track_factory: callable that returns a VideoStreamTrack.
    """
    pc = RTCPeerConnection()
    track: Optional[Any] = None

    async def send(msg: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception as e:
            logger.warning("Send to client failed: %s", e)

    @pc.on("icecandidate")
    def on_ice_candidate(candidate: Any) -> None:
        if candidate is not None:
            asyncio.get_running_loop().create_task(
                send(build_ice_message(ice_candidate_to_dict(candidate)))
            )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in ("failed", "closed"):
            if track is not None:
                track.stop()
            await pc.close()

    try:
        track = track_factory()
        pc.addTrack(track)

        while True:
            raw = await websocket.receive_text()
            msg_type, msg = parse_message(raw)
            if msg_type is None:
                continue

            if msg_type == "offer":
                sdp = msg.get("sdp")
                if not sdp:
                    continue
                offer = RTCSessionDescription(sdp=sdp, type=msg_type)
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await send(build_answer_message(pc.localDescription))

            elif msg_type == "ice" and "candidate" in msg:
                rtc_cand = parse_ice_candidate_from_message(msg["candidate"])
                try:
                    await pc.addIceCandidate(rtc_cand)
                except Exception as e:
                    logger.warning("addIceCandidate failed: %s", e)

    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        if track is not None:
            track.stop()
        await pc.close()
