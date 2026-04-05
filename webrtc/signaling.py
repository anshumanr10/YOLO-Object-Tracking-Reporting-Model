"""Signaling JSON parse/build (offer / answer / ICE)."""

from __future__ import annotations

import json
from typing import Any


def parse_message(raw: str) -> tuple[str | None, dict[str, Any]]:
    """Return ``(type, full_message)`` or ``(None, {})`` if invalid."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None, {}
    if not isinstance(msg, dict):
        return None, {}
    return msg.get("type"), msg


def build_answer_message(local_description: Any) -> dict[str, Any]:
    if local_description is None:
        return {}
    return {"type": local_description.type, "sdp": local_description.sdp}


def build_ice_message(candidate_dict: dict[str, Any]) -> dict[str, Any]:
    return {"type": "ice", "candidate": candidate_dict}
