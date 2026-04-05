"""ICE candidate JSON ↔ aiortc (browser signaling shape)."""

from __future__ import annotations

from typing import Any

from aiortc.sdp import candidate_to_sdp


def ice_candidate_to_dict(candidate: Any) -> dict[str, Any]:
    """Serialize server ICE candidate for the browser (``candidate:`` SDP fragment)."""
    sdp = "candidate:" + candidate_to_sdp(candidate)
    return {
        "candidate": sdp,
        "sdpMid": getattr(candidate, "sdpMid", None),
        "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None),
    }


def parse_ice_candidate_from_message(msg: dict[str, Any]) -> Any | None:
    """Build ``RTCIceCandidate`` from browser ``RTCIceCandidateInit`` dict, or ``None``."""
    from aiortc import RTCIceCandidate
    from aiortc.sdp import candidate_from_sdp

    cand = msg if isinstance(msg, dict) else {}
    sdp = cand.get("candidate")
    if sdp is None:
        return None
    sdp = sdp.strip() if isinstance(sdp, str) else ""
    if sdp.lower() in ("", "null", "none"):
        return None
    if sdp.startswith("a=candidate:"):
        sdp = sdp[len("a=candidate:") :]
    if sdp.startswith("candidate:"):
        sdp = sdp[len("candidate:") :]
    rtc_cand = candidate_from_sdp(sdp)
    sdp_mid = cand.get("sdpMid")
    sdp_mline = cand.get("sdpMLineIndex")
    if sdp_mid is not None or sdp_mline is not None:
        rtc_cand = RTCIceCandidate(
            component=rtc_cand.component,
            foundation=rtc_cand.foundation,
            ip=rtc_cand.ip,
            port=rtc_cand.port,
            priority=rtc_cand.priority,
            protocol=rtc_cand.protocol,
            type=rtc_cand.type,
            relatedAddress=getattr(rtc_cand, "relatedAddress", None),
            relatedPort=getattr(rtc_cand, "relatedPort", None),
            sdpMid=sdp_mid,
            sdpMLineIndex=sdp_mline,
        )
    return rtc_cand
