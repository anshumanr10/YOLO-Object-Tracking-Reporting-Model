"""Video codec preference order for low-delay negotiation (H.264 then VP8, then rest)."""

from __future__ import annotations

from typing import Any

from aiortc.codecs import is_rtx
from aiortc import RTCPeerConnection
from aiortc.rtcrtpsender import RTCRtpSender


def _lan_video_codec_preference_list() -> list[Any]:
    caps = RTCRtpSender.getCapabilities("video")
    base = [c for c in caps.codecs if not is_rtx(c)]
    h264 = [c for c in base if c.mimeType.lower() == "video/h264"]
    vp8 = [c for c in base if c.mimeType.lower() == "video/vp8"]
    rest = [c for c in base if c not in h264 + vp8]
    return h264 + vp8 + rest


def apply_lan_video_codec_preferences(pc: RTCPeerConnection) -> None:
    """
    Prefer H.264 (often hardware) then VP8 on the sending video transceiver.

    Call after ``setRemoteDescription(offer)`` and before ``createAnswer()``.
    """
    prefs = _lan_video_codec_preference_list()
    if not prefs:
        return
    for transceiver in pc.getTransceivers():
        if transceiver.kind != "video":
            continue
        st = transceiver.sender
        if st.track is None:
            continue
        transceiver.setCodecPreferences(prefs)
        return
