"""WebSocket signaling + LAN ``RTCPeerConnection`` + one outbound video track."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription

from webrtc.ice import ice_candidate_to_dict, parse_ice_candidate_from_message
from webrtc.lan_config import rtc_configuration_lan
from webrtc.signaling import build_answer_message, build_ice_message, parse_message
from webrtc.video_codecs import apply_lan_video_codec_preferences

logger = logging.getLogger(__name__)


def _create_lan_peer_connection() -> RTCPeerConnection:
    return RTCPeerConnection(rtc_configuration_lan())


async def _send_text(websocket: Any, obj: dict[str, Any]) -> None:
    try:
        await websocket.send_text(json.dumps(obj))
    except Exception as e:
        logger.warning("webrtc send failed: %s", e)


async def _handle_offer(
    pc: RTCPeerConnection,
    websocket: Any,
    sdp: str,
) -> None:
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
    apply_lan_video_codec_preferences(pc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await _send_text(websocket, build_answer_message(pc.localDescription))


async def _handle_client_ice(pc: RTCPeerConnection, candidate_payload: Any) -> None:
    if not isinstance(candidate_payload, dict):
        return
    rtc_cand = parse_ice_candidate_from_message(candidate_payload)
    await pc.addIceCandidate(rtc_cand)


async def handle_webrtc_connection(
    websocket: Any,
    track_factory: Callable[[], Any],
) -> None:
    """
    Signaling: JSON ``offer`` / ``ice`` in; ``answer`` / ``ice`` out. Media: one added track.

    Contract: ``websocket`` supports ``send_text`` / ``receive_text``; ``track_factory`` returns an
    aiortc ``VideoStreamTrack``. Optimized for same-LAN browser peers (no STUN/TURN).
    """
    pc = _create_lan_peer_connection()
    track: Optional[Any] = None
    loop = asyncio.get_running_loop()

    @pc.on("icecandidate")
    def _on_icecandidate(candidate: Any) -> None:
        if candidate is None:
            return
        loop.create_task(_send_text(websocket, build_ice_message(ice_candidate_to_dict(candidate))))

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:
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
                if sdp:
                    await _handle_offer(pc, websocket, sdp)
            elif msg_type == "ice" and "candidate" in msg:
                try:
                    await _handle_client_ice(pc, msg["candidate"])
                except Exception as e:
                    logger.warning("addIceCandidate failed: %s", e)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        if track is not None:
            track.stop()
        await pc.close()
