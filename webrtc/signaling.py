"""
WebRTC signaling message parsing and building.
Deals only with message shape and JSON; no WebSocket or RTCPeerConnection.
"""

import json
from typing import Any


def parse_message(raw: str) -> tuple[str | None, dict[str, Any]]:
    """
    Parse a signaling JSON message.

    Returns (msg_type, payload) where msg_type is "offer", "ice", or None,
    and payload is the full parsed dict. Invalid JSON or missing type yields (None, {}).
    """
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None, {}
    if not isinstance(msg, dict):
        return None, {}
    return msg.get("type"), msg


def build_answer_message(local_description: Any) -> dict[str, Any]:
    """Build the JSON object to send for an SDP answer."""
    if local_description is None:
        return {}
    return {
        "type": local_description.type,
        "sdp": local_description.sdp,
    }


def build_ice_message(candidate_dict: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON object to send for an ICE candidate."""
    return {"type": "ice", "candidate": candidate_dict}
