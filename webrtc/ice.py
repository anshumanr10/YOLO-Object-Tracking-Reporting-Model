"""
ICE candidate serialization and parsing for WebRTC signaling.
Browser and server exchange candidates as JSON; this module converts
to/from aiortc RTCIceCandidate.
"""

from typing import Any

from aiortc.sdp import candidate_to_sdp


def ice_candidate_to_dict(candidate: Any) -> dict[str, Any]:
    """
    Serialize aiortc RTCIceCandidate for the browser.

    Browser-side RTCIceCandidateInit.candidate expects a string that starts with
    "candidate:" (NOT "a=candidate:").
    """
    sdp = "candidate:" + candidate_to_sdp(candidate)
    return {
        "candidate": sdp,
        "sdpMid": getattr(candidate, "sdpMid", None),
        "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None),
    }


def parse_ice_candidate_from_message(msg: dict[str, Any]) -> Any | None:
    """
    Parse client ICE candidate from a signaling message payload.

    msg is the "candidate" object from {"type": "ice", "candidate": {...}}.
    Returns an RTCIceCandidate for pc.addIceCandidate(), or None to signal
    end-of-candidates (addIceCandidate(None)).
    """
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
